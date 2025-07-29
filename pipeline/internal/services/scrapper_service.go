package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"fmt"
	"github.com/gocolly/colly/v2"
	"github.com/gocolly/colly/v2/debug"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

type ScraperService struct {
	collector *colly.Collector
	logger    *logger.Logger
	config    *config.EtcConfig

	rateLimiter chan struct{}
	mu          sync.RWMutex

	userAgents []string
	uaIndex    int
}

type ScrapedContent struct {
	URL         string            `json:"url"`
	Title       string            `json:"title"`
	Content     string            `json:"content"`
	Description string            `json:"description"`
	Author      string            `json:"author"`
	PublishedAt time.Time         `json:"published_at"`
	ImageURL    string            `json:"image_url"`
	Tags        []string          `json:"tags"`
	Metadata    map[string]string `json:"metadata"`
	ScrapedAt   time.Time         `json:"scraped_at"`
	Success     bool              `json:"success"`
	Error       string            `json:"error"`
}

type ScrapingRequest struct {
	URLs            []string          `json:"urls"`
	MaxConcurrency  int               `json:"max_concurrency"`
	Timeout         time.Duration     `json:"timeout"`
	RetryAttempts   int               `json:"retry_attempts"`
	CustomSelectors map[string]string `json:"custom_selectors"`
	Headers         map[string]string `json:"headers"`
}

type ScrapingResult struct {
	SuccessfulScrapes []ScrapedContent `json:"successful_scrapes"`
	FailedScrapes     []ScrapedContent `json:"failed_scrapes"`
	TotalRequested    int              `json:"total_requested"`
	TotalSuccessful   int              `json:"total_successful"`
	TotalFailed       int              `json:"total_failed"`
	Duration          time.Duration    `json:"duration"`
}

func NewScraperService(config *config.EtcConfig, logger *logger.Logger) (*ScraperService, error) {
	collector := colly.NewCollector(
		colly.Debugger(&debug.LogDebugger{}),
		colly.UserAgent("Anya-AI-News-Assistant/1.0 (+https://anya-ai.com/bot)"),
	)

	collector.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: 3,
		Delay:       5 * time.Second,
	})

	collector.SetRequestTimeout(30 * time.Second)

	userAgents := []string{
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
	}

	service := &ScraperService{
		collector:   collector,
		logger:      logger,
		config:      config,
		rateLimiter: make(chan struct{}, 5), // Limit to 5 concurrent scrapes
		userAgents:  userAgents,
		uaIndex:     0,
	}

	service.setupCallbacks()

	if err := service.testConnection(); err != nil {
		return nil, fmt.Errorf("Scraper Service testConnection failed %v", err)
	}

	logger.Info("Scraper Service test connection successfully", "rate_limit", "5 concurrent requests", "delay", "2 seconds between requests", "timeout", "30 seconds")

	return service, nil

}

func (service *ScraperService) testConnection() error {
	testURL := "https://httpbin.org/user-agent"

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := service.ScrapeURL(ctx, testURL)
	if err != nil {
		return fmt.Errorf("Scraper Service testConnection failed %v", err)
	}

	service.logger.Info("scraper service test successfull")
	return nil
}

func (service *ScraperService) ScrapeURL(ctx context.Context, targetURL string) (*ScrapedContent, error) {
	if targetURL == "" {
		return nil, fmt.Errorf("Target URL cannot be empty")
	}

	startTime := time.Now()

	if _, err := url.Parse(targetURL); err != nil {
		return nil, fmt.Errorf("Target URL parsing failed %v", err)
	}

	service.logger.LogService("scraper", "scraper_url", 0, map[string]interface{}{
		"url": targetURL,
	}, nil)

	select {
	case service.rateLimiter <- struct{}{}:
		defer func() { <-service.rateLimiter }()
	case <-ctx.Done():
		return nil, models.NewTimeoutError("SCRAPER_TIMEOUT", "Scraping request timed out").WithCause(ctx.Err())
	}

	content := &ScrapedContent{
		URL:       targetURL,
		ScrapedAt: time.Now(),
		Success:   false,
		Metadata:  make(map[string]string),
	}

	c := service.collector.Clone()

	var scrapingError error

	c.OnHTML("html", func(e *colly.HTMLElement) {
		content.Content = service.extractArticleContent(e)
		content.Title = service.extractTitle(e)
		content.Description = service.extractDescription(e)
		content.Author = service.extractAuthor(e)
		content.PublishedAt = service.extractPublishedDate(e)
		content.ImageURL = service.extractMainImage(e)
		content.Tags = service.extractTags(e)

		content.Metadata["lang"] = e.Attr("lang")
		content.Metadata["charset"] = e.Attr("charset")

		content.Success = content.Content != "" || content.Title != ""

	})

	c.OnError(func(r *colly.Response, err error) {
		scrapingError = err
		content.Error = err.Error()
	})

	done := make(chan bool)
	go func() {
		defer func() { done <- true }()
		err := c.Visit(targetURL)
		if err != nil {
			scrapingError = err
			content.Error = err.Error()
		}
	}()

	select {
	case <-done:
	// scrapping done
	case <-ctx.Done():
		return nil, models.NewTimeoutError("SCRAPER_TIMEOUT", "Scraping Request Time out").WithCause(ctx.Err())
	}

	if scrapingError != nil && !content.Success {
		service.logger.LogService("scraper", "scrape_url", time.Since(startTime), map[string]interface{}{
			"url": targetURL,
		}, scrapingError)
		content.Error = scrapingError.Error()
		return content, nil
	}

	content.Content = service.cleanContent(content.Content)
	content.Description = service.cleanContent(content.Description)

	duration := time.Since(startTime)
	service.logger.LogService("scraper", "scraper_url", duration, map[string]interface{}{
		"url":            targetURL,
		"success":        content.Success,
		"content_length": len(content.Content),
		"title":          content.Title != "",
		"description":    content.Description != "",
	}, nil)

	return content, nil
}

func (service *ScraperService) ScrapeMultipleURLs(ctx context.Context, req *ScrapingRequest) (*ScrapingResult, error) {
	if len(req.URLs) == 0 {
		return nil, fmt.Errorf("Scraping Request URLs cannot be empty")
	}

	startTime := time.Now()

	service.logger.LogService("scrapper", "scrape_multiple_urls", 0, map[string]interface{}{
		"url_count":       len(req.URLs),
		"max_concurrency": req.MaxConcurrency,
	}, nil)

	result := &ScrapingResult{
		TotalRequested:    len(req.URLs),
		SuccessfulScrapes: []ScrapedContent{},
		FailedScrapes:     []ScrapedContent{},
	}

	if req.MaxConcurrency == 0 {
		req.MaxConcurrency = 3
	}
	if req.Timeout == 0 {
		req.Timeout = 30 * time.Second
	}

	semaphore := make(chan struct{}, req.MaxConcurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, targetURL := range req.URLs {
		wg.Add(1)
		go func(url string) {
			defer wg.Done()

			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				return
			}

			requestCtx, cancel := context.WithTimeout(ctx, req.Timeout)
			defer cancel()

			content, err := service.ScrapeURL(requestCtx, targetURL)
			mu.Lock()
			if err != nil || !content.Success {
				result.FailedScrapes = append(result.FailedScrapes, *content)
				result.TotalFailed++
			} else {
				result.SuccessfulScrapes = append(result.SuccessfulScrapes, *content)
				result.TotalSuccessful++
			}

			mu.Unlock()

		}(targetURL)

	}

	wg.Wait()

	result.Duration = time.Since(startTime)

	service.logger.LogService("scraper", "scrape_multiple_urls", result.Duration, map[string]interface{}{
		"total_requested": result.TotalRequested,
		"total_failed":    result.TotalFailed,
		"total_success":   result.TotalSuccessful,
		"success_rate":    float64(result.TotalSuccessful) / float64(result.TotalRequested) * 100,
	}, nil)

	return result, nil

}

func (service *ScraperService) ScrapeNewsArticle(ctx context.Context, article *models.NewsArticle) (*models.NewsArticle, error) {
	if article == nil || article.URL == "" {
		return nil, fmt.Errorf("Article url cannot be empty")
	}

	content, err := service.ScrapeURL(ctx, article.URL)
	if err != nil {
		return nil, fmt.Errorf("Scraper Service failed to scrape :  %v", err)
	}

	if content.Success {
		if content.Content != "" {
			article.Content = content.Content
		}
		if content.Author != "" && article.Author == "" {
			article.Author = content.Author
		}
		if content.Description != "" && article.Description == "" {
			article.Description = content.Description
		}
		if content.ImageURL != "" && article.ImageURL == "" {
			article.ImageURL = content.ImageURL
		}
	}

	return article, nil

}

func (service *ScraperService) extractArticleContent(e *colly.HTMLElement) string {
	selectors := []string{
		"article",
		"[role='main'] article",
		".article-content",
		".post-content",
		".entry-content",
		".content-body",
		".story-body",
		".article-body",
		"main article",
		"[itemtype*='Article']",
		".post-body",
		".entry",
	}

	for _, selector := range selectors {
		if content := e.ChildText(selector); content != "" {
			return content
		}
	}

	paragraphs := e.ChildTexts("p")
	if len(paragraphs) > 0 {
		return strings.Join(paragraphs, "\n\n")
	}

	return ""

}

func (service *ScraperService) extractTitle(e *colly.HTMLElement) string {
	selectors := []string{
		"h1.article-title",
		"h1.entry-title",
		"ht.post-title",
		".article-header h1",
		"article h2",
		"h1",
	}

	for _, selector := range selectors {
		if title := e.ChildText(selector); title != "" {
			return strings.TrimSpace(title)
		}
	}

	if title := e.ChildText("title"); title != "" {
		return strings.TrimSpace(title)
	}

	return ""

}

func (service *ScraperService) extractDescription(e *colly.HTMLElement) string {
	if desc := e.Attr("meta[name='description']"); desc != "" {
		return strings.TrimSpace(desc)
	}

	// Try Open Graph description
	if desc := e.Attr("meta[property='og:description']"); desc != "" {
		return strings.TrimSpace(desc)
	}

	// Try Twitter description
	if desc := e.Attr("meta[name='twitter:description']"); desc != "" {
		return strings.TrimSpace(desc)
	}

	// Try first paragraph
	if desc := e.ChildText("article p:first-of-type"); desc != "" {
		return strings.TrimSpace(desc)
	}

	return ""
}

func (service *ScraperService) extractPublishedDate(e *colly.HTMLElement) time.Time {
	if dateStr := e.Attr("[itemprop='datePublished']"); dateStr != "" {
		if date, err := time.Parse(time.RFC3339, dateStr); err == nil {
			return date
		}
	}

	if dateStr := e.Attr("time"); dateStr != "" {
		if date, err := time.Parse(time.RFC3339, dateStr); err == nil {
			return date
		}
	}

	metaSelectors := []string{
		"meta[property='article:published_time']",
		"meta[name='publish-date']",
		"meta[name='date']",
	}

	for _, selector := range metaSelectors {
		if dateStr := e.Attr(selector); dateStr != "" {
			if date, err := time.Parse(time.RFC3339, dateStr); err == nil {
				return date
			}
		}
	}

	return time.Time{}
}

func (service *ScraperService) extractMainImage(e *colly.HTMLElement) string {
	// Try Open Graph image
	if img := e.Attr("meta[property='og:image']"); img != "" {
		return img
	}

	// Try Twitter image
	if img := e.Attr("meta[name='twitter:image']"); img != "" {
		return img
	}

	// Try article images
	selectors := []string{
		"article img:first-of-type",
		".article-image img",
		".featured-image img",
		".post-image img",
	}

	for _, selector := range selectors {
		if img := e.ChildAttr(selector, "src"); img != "" {
			return img
		}
	}

	return ""
}

func (service *ScraperService) extractTags(e *colly.HTMLElement) []string {
	var tags []string

	// Try meta keywords
	if keywords := e.Attr("meta[name='keywords']"); keywords != "" {
		tags = append(tags, strings.Split(keywords, ",")...)
	}

	// Try article tags
	tagSelectors := []string{
		".tags a",
		".article-tags a",
		".post-tags a",
		"[rel='tag']",
	}

	for _, selector := range tagSelectors {
		e.ForEach(selector, func(_ int, el *colly.HTMLElement) {
			if tag := strings.TrimSpace(el.Text); tag != "" {
				tags = append(tags, tag)
			}
		})
	}

	// Clean and deduplicate tags
	return service.cleanTags(tags)
}

func (service *ScraperService) extractCharset(e *colly.HTMLElement) string {
	if charset := e.Attr("meta[charset]"); charset != "" {
		return charset
	}
	if charset := e.Attr("meta[http-equiv='Content-Type']"); charset != "" {
		// Parse charset from content-type
		if strings.Contains(charset, "charset=") {
			parts := strings.Split(charset, "charset=")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "utf-8"
}

func (service *ScraperService) cleanContent(content string) string {
	if content == "" {
		return content
	}

	// Remove excessive whitespace
	re := regexp.MustCompile(`\s+`)
	content = re.ReplaceAllString(content, " ")

	// Trim
	content = strings.TrimSpace(content)

	// Limit length
	if len(content) > 10000 {
		content = content[:10000] + "..."
	}

	return content
}

func (service *ScraperService) cleanTags(tags []string) []string {
	seen := make(map[string]bool)
	var cleaned []string

	for _, tag := range tags {
		tag = strings.TrimSpace(tag)
		tag = strings.ToLower(tag)
		if tag != "" && !seen[tag] && len(tag) > 2 {
			seen[tag] = true
			cleaned = append(cleaned, tag)
		}
	}

	return cleaned
}

func (service *ScraperService) extractAuthor(e *colly.HTMLElement) string {
	selectors := []string{
		"[rel='author']",
		".author-name",
		".article-author",
		".byline-author",
		"[itemprop='author']",
		".post-author",
	}

	for _, selector := range selectors {
		if author := e.ChildText(selector); author != "" {
			return strings.TrimSpace(author)
		}
	}

	// Try meta tags
	if author := e.Attr("meta[name='author']"); author != "" {
		return strings.TrimSpace(author)
	}

	return ""
}

func (service *ScraperService) setupCallbacks() {
	service.collector.OnRequest(func(r *colly.Request) {
		service.mu.Lock()
		r.Headers.Set("User-Agent", service.userAgents[service.uaIndex])
		service.uaIndex = (service.uaIndex + 1) % len(service.userAgents)
		service.mu.Unlock()

		r.Headers.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
		r.Headers.Set("Accept-Language", "en-US,en;q=0.5")
		r.Headers.Set("Accept-Encoding", "gzip, deflate")
		r.Headers.Set("Connection", "keep-alive")
		r.Headers.Set("Upgrade-Insecure-Requests", "1")

		service.logger.Debug("Starting scrape request", "url", r.URL.String())

		service.collector.OnResponse(func(r *colly.Response) {
			service.logger.Debug(" scrape response received", "url", r.Request.URL.String(), "status", r.StatusCode, "size", len(r.Body))
		})

		service.collector.OnError(func(r *colly.Response, err error) {
			service.logger.WithFields(logger.Fields{
				"request_url": r.Request.URL.String(),
				"error":       err.Error(),
				"status":      r.StatusCode,
			})
		})

	})
}

func (service *ScraperService) HealthCheck(ctx context.Context) error {
	// Test with a simple, reliable URL
	testURL := "https://httpbin.org/html"

	testCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	content, err := service.ScrapeURL(testCtx, testURL)
	if err != nil {
		return fmt.Errorf("health check scrape failed: %w", err)
	}

	if !content.Success {
		return fmt.Errorf("health check scrape was unsuccessful")
	}

	return nil
}
