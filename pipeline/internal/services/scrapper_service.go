package services

import (
	"Infiya-ai-pipeline/internal/config"
	"Infiya-ai-pipeline/internal/models"
	"Infiya-ai-pipeline/internal/pkg/logger"
	"context"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gocolly/colly/v2"
	"github.com/gocolly/colly/v2/debug"
)

type ScraperService struct {
	collector   *colly.Collector
	logger      *logger.Logger
	config      *config.ScraperConfig
	rateLimiter chan struct{}
	mu          sync.RWMutex
	userAgents  []string
	uaIndex     int
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

func NewScraperService(config config.ScraperConfig, logger *logger.Logger) (*ScraperService, error) {
	collector := colly.NewCollector(
		colly.Debugger(&debug.LogDebugger{}),
		colly.UserAgent("Infiya-AI-News-Assistant/1.0 (+https://infiya-ai.com/bot)"),
		colly.AllowedDomains(), // Allow all domains
	)

	collector.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: 2,
		Delay:       3 * time.Second,
	})

	collector.SetRequestTimeout(60 * time.Second)

	userAgents := []string{
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/120.0",
		"Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/120.0",
	}

	service := &ScraperService{
		collector:   collector,
		logger:      logger,
		config:      &config,
		rateLimiter: make(chan struct{}, 5),
		userAgents:  userAgents,
		uaIndex:     0,
	}

	service.setupCallbacks()
	logger.Info("Infiya Scraper Service initialized successfully",
		"rate_limit", "5 concurrent requests",
		"delay", "3 seconds between requests",
		"timeout", "60 seconds",
		"content_extraction", "p-tag-focused")

	return service, nil
}

func (service *ScraperService) ScrapeURL(ctx context.Context, targetURL string) (*ScrapedContent, error) {
	startTime := time.Now()

	content := &ScrapedContent{
		URL:       targetURL,
		ScrapedAt: time.Now(),
		Success:   false,
		Metadata:  make(map[string]string),
		Error:     "",
		Tags:      []string{},
	}

	if targetURL == "" {
		content.Error = "Target URL cannot be empty"
		return content, fmt.Errorf("Target URL cannot be empty")
	}

	parsedURL, err := url.Parse(targetURL)
	if err != nil {
		content.Error = fmt.Sprintf("Invalid URL format: %v", err)
		service.logger.Error("URL parsing failed", "url", targetURL, "error", err)
		return content, fmt.Errorf("Target URL parsing failed %v", err)
	}

	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		content.Error = fmt.Sprintf("Unsupported URL scheme: %s", parsedURL.Scheme)
		return content, fmt.Errorf("Unsupported URL scheme: %s", parsedURL.Scheme)
	}

	service.logger.Info("Starting P-tag focused scrape",
		"url", targetURL,
		"domain", parsedURL.Host,
		"scheme", parsedURL.Scheme)

	// Rate limiting with context
	select {
	case service.rateLimiter <- struct{}{}:
		defer func() { <-service.rateLimiter }()
	case <-ctx.Done():
		content.Error = "Rate limiter timeout"
		return content, models.NewTimeoutError("SCRAPER_TIMEOUT", "Rate limiter timeout").WithCause(ctx.Err())
	}

	c := service.collector.Clone()
	var scrapingError error
	var httpStatusCode int
	var responseSize int
	var contentProcessed bool

	c.OnRequest(func(r *colly.Request) {
		service.mu.Lock()
		userAgent := service.userAgents[service.uaIndex]
		service.uaIndex = (service.uaIndex + 1) % len(service.userAgents)
		service.mu.Unlock()

		r.Headers.Set("User-Agent", userAgent)
		r.Headers.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
		r.Headers.Set("Accept-Language", "en-US,en;q=0.5")
		r.Headers.Set("Accept-Encoding", "gzip, deflate, br")
		r.Headers.Set("DNT", "1")
		r.Headers.Set("Connection", "keep-alive")
		r.Headers.Set("Upgrade-Insecure-Requests", "1")
		r.Headers.Set("Sec-Fetch-Dest", "document")
		r.Headers.Set("Sec-Fetch-Mode", "navigate")
		r.Headers.Set("Sec-Fetch-Site", "none")
		r.Headers.Set("Cache-Control", "max-age=0")

		service.logger.Debug("P-tag scraper request sent",
			"url", r.URL.String(),
			"user_agent", userAgent[:50]+"...")
	})

	c.OnResponse(func(r *colly.Response) {
		httpStatusCode = r.StatusCode
		responseSize = len(r.Body)

		service.logger.Info("Scraper response received",
			"url", r.Request.URL.String(),
			"status", r.StatusCode,
			"size", responseSize,
			"content_type", r.Headers.Get("Content-Type"))

		content.Metadata["status_code"] = fmt.Sprintf("%d", r.StatusCode)
		content.Metadata["content_type"] = r.Headers.Get("Content-Type")
		content.Metadata["response_size"] = fmt.Sprintf("%d", responseSize)
	})

	c.OnHTML("html", func(e *colly.HTMLElement) {
		if e == nil {
			service.logger.Warn("HTML element is nil", "url", targetURL)
			return
		}

		contentProcessed = true
		service.logger.Debug("Processing HTML content with P-tag extraction",
			"url", targetURL,
			"html_length", len(e.Text))

		// P-TAG FOCUSED EXTRACTION - This is the key change!
		content.Content = service.extractArticleContentFromParagraphs(e)
		content.Title = service.extractTitle(e)
		content.Description = service.extractDescription(e)
		content.Author = service.extractAuthor(e)
		content.PublishedAt = service.extractPublishedDate(e)
		content.ImageURL = service.extractMainImage(e)
		content.Tags = service.extractTags(e)
		content.Metadata["lang"] = e.Attr("lang")
		content.Metadata["charset"] = service.extractCharset(e)

		hasTitle := strings.TrimSpace(content.Title) != ""
		hasContent := strings.TrimSpace(content.Content) != ""
		hasDescription := strings.TrimSpace(content.Description) != ""

		content.Success = hasTitle || hasContent || hasDescription

		service.logger.Info("P-tag extraction results",
			"url", targetURL,
			"has_title", hasTitle,
			"has_content", hasContent,
			"has_description", hasDescription,
			"title", safeTruncate(content.Title, 50),
			"content_length", len(content.Content),
			"paragraph_count", strings.Count(content.Content, "\n\n")+1,
			"success", content.Success)
	})

	c.OnError(func(r *colly.Response, err error) {
		scrapingError = err
		if r != nil {
			httpStatusCode = r.StatusCode
		}

		service.logger.Error("Scraper error occurred",
			"url", targetURL,
			"error", err.Error(),
			"status_code", httpStatusCode,
			"response_size", len(r.Body))

		content.Error = fmt.Sprintf("HTTP %d: %s", httpStatusCode, err.Error())
		content.Metadata["error_type"] = "scraping_error"
	})

	done := make(chan bool, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				service.logger.Error("Panic in scraper goroutine", "panic", r, "url", targetURL)
				scrapingError = fmt.Errorf("scraper panic: %v", r)
				content.Error = fmt.Sprintf("Scraper panic: %v", r)
			}
			select {
			case done <- true:
			default:
			}
		}()

		err := c.Visit(targetURL)
		if err != nil {
			scrapingError = err
			content.Error = err.Error()
			service.logger.Error("Visit failed", "url", targetURL, "error", err)
		}
	}()

	select {
	case <-done:
		service.logger.Debug("P-tag scraping completed", "url", targetURL)
	case <-ctx.Done():
		content.Error = "Context timeout"
		content.Success = false
		service.logger.Warn("Scraping timed out", "url", targetURL, "duration", time.Since(startTime))
		return content, models.NewTimeoutError("SCRAPER_TIMEOUT", "Scraping request timed out").WithCause(ctx.Err())
	}

	if !contentProcessed && scrapingError == nil {
		service.logger.Warn("No HTML content processed", "url", targetURL, "status", httpStatusCode)
		content.Error = fmt.Sprintf("No HTML content found (HTTP %d)", httpStatusCode)
	}

	// Final content cleaning
	content.Content = service.cleanContent(content.Content)
	content.Description = service.cleanContent(content.Description)
	content.Title = strings.TrimSpace(content.Title)

	duration := time.Since(startTime)
	service.logger.LogService("scraper", "scraper_url_ptag", duration, map[string]interface{}{
		"url":            targetURL,
		"success":        content.Success,
		"content_length": len(content.Content),
		"title":          content.Title != "",
		"description":    content.Description != "",
		"status_code":    httpStatusCode,
		"response_size":  responseSize,
		"error":          content.Error,
	}, scrapingError)

	return content, nil
}

// ================ P-TAG FOCUSED CONTENT EXTRACTION ================

// extractArticleContentFromParagraphs - NEW METHOD focuses specifically on P tags
func (service *ScraperService) extractArticleContentFromParagraphs(e *colly.HTMLElement) string {
	var validParagraphs []string

	service.logger.Debug("Starting P-tag focused extraction")

	// Extract text from all P tags, including nested formatting tags
	e.ForEach("p", func(i int, p *colly.HTMLElement) {
		// Get all text within the p tag, including nested tags like <strong>, <b>, <em>, etc.
		text := strings.TrimSpace(p.Text)

		// Quality filters for paragraphs
		if service.isValidParagraph(text) {
			validParagraphs = append(validParagraphs, text)
		}
	})

	// If we don't have enough good paragraphs, try paragraphs within article/main containers
	if len(validParagraphs) < 3 {
		service.logger.Debug("Low paragraph count, trying container-specific P tags", "count", len(validParagraphs))

		containerSelectors := []string{"article", "main", ".article-content", ".post-content", ".entry-content", ".content"}

		for _, selector := range containerSelectors {
			e.ForEach(selector+" p", func(i int, p *colly.HTMLElement) {
				text := strings.TrimSpace(p.Text)
				if service.isValidParagraph(text) && !service.containsText(validParagraphs, text) {
					validParagraphs = append(validParagraphs, text)
				}
			})

			if len(validParagraphs) >= 3 {
				break
			}
		}
	}

	content := strings.Join(validParagraphs, "\n\n")
	content = service.cleanContent(content)

	service.logger.Debug("P-tag extraction completed",
		"paragraphs_found", len(validParagraphs),
		"final_length", len(content))

	return content
}

// isValidParagraph checks if a paragraph contains valid content
func (service *ScraperService) isValidParagraph(text string) bool {
	// Minimum length check
	if len(text) < 50 {
		return false
	}

	// Maximum length check (probably not a paragraph if too long)
	if len(text) > 2000 {
		return false
	}

	// Check for noise patterns
	if service.isNoiseText(text) {
		return false
	}

	// Check word count (good paragraphs have multiple words)
	wordCount := len(strings.Fields(text))
	if wordCount < 8 {
		return false
	}

	// Check for excessive capitals (likely headers/navigation)
	upperCount := 0
	for _, r := range text {
		if r >= 'A' && r <= 'Z' {
			upperCount++
		}
	}
	if len(text) > 0 && float64(upperCount)/float64(len(text)) > 0.5 {
		return false
	}

	return true
}

// containsText checks if text is already in the slice (avoid duplicates)
func (service *ScraperService) containsText(paragraphs []string, text string) bool {
	for _, p := range paragraphs {
		if p == text {
			return true
		}
	}
	return false
}

// isNoiseText checks if text contains common noise patterns
func (service *ScraperService) isNoiseText(text string) bool {
	lowerText := strings.ToLower(text)

	noisePatterns := []string{
		"subscribe", "newsletter", "advertisement", "cookie",
		"privacy policy", "terms of service", "follow us",
		"share this", "related articles", "read more",
		"click here", "sign up", "log in", "contact us",
		"trending now", "most popular", "you might also like",
	}

	for _, pattern := range noisePatterns {
		if strings.Contains(lowerText, pattern) {
			return true
		}
	}

	return false
}

// ================ ENHANCED CONTENT CLEANING ================

// cleanContent performs aggressive content cleaning
func (service *ScraperService) cleanContent(content string) string {
	if content == "" {
		return content
	}

	// Step 1: Normalize whitespace
	re := regexp.MustCompile(`\s+`)
	content = re.ReplaceAllString(content, " ")

	// Step 2: Remove noise patterns
	patterns := []string{
		`(?i)javascript:void\(0\)`,
		`(?i)advertisement`,
		`(?i)sponsored content`,
		`(?i)subscribe to.*newsletter`,
		`(?i)follow us on`,
		`(?i)share this article`,
		`(?i)read more articles?`,
		`(?i)click here for more`,
		`(?i)privacy policy`,
		`(?i)terms of service`,
		`(?i)all rights reserved`,
		`(?i)copyright \d+`,
		`(?i)powered by`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		content = re.ReplaceAllString(content, "")
	}

	// Step 3: Clean formatting
	re = regexp.MustCompile(`[.]{3,}`)
	content = re.ReplaceAllString(content, "...")

	re = regexp.MustCompile(`[-]{3,}`)
	content = re.ReplaceAllString(content, "---")

	// Step 4: Final cleanup
	content = strings.TrimSpace(content)

	return content
}

// ================ REMAINING METHODS FROM ORIGINAL ================

func (service *ScraperService) ScrapeMultipleURLs(ctx context.Context, req *ScrapingRequest) (*ScrapingResult, error) {
	if req == nil {
		return nil, fmt.Errorf("Scraping request cannot be nil")
	}
	if len(req.URLs) == 0 {
		return nil, fmt.Errorf("Scraping Request URLs cannot be empty")
	}

	startTime := time.Now()
	service.logger.Info("Starting P-tag focused multiple URL scraping",
		"url_count", len(req.URLs),
		"max_concurrency", req.MaxConcurrency,
		"timeout", req.Timeout)

	result := &ScrapingResult{
		TotalRequested:    len(req.URLs),
		SuccessfulScrapes: make([]ScrapedContent, 0),
		FailedScrapes:     make([]ScrapedContent, 0),
		TotalSuccessful:   0,
		TotalFailed:       0,
	}

	if req.MaxConcurrency == 0 {
		req.MaxConcurrency = 3
	}
	if req.Timeout == 0 {
		req.Timeout = 60 * time.Second
	}

	semaphore := make(chan struct{}, req.MaxConcurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, targetURL := range req.URLs {
		wg.Add(1)
		go func(url string, index int) {
			defer func() {
				if r := recover(); r != nil {
					service.logger.Error("Panic in scraper goroutine", "panic", r, "url", url, "index", index)
					mu.Lock()
					result.FailedScrapes = append(result.FailedScrapes, ScrapedContent{
						URL:       url,
						ScrapedAt: time.Now(),
						Success:   false,
						Error:     fmt.Sprintf("Scraper panic: %v", r),
						Metadata:  make(map[string]string),
						Tags:      []string{},
					})
					result.TotalFailed++
					mu.Unlock()
				}
				wg.Done()
			}()

			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				return
			}

			requestCtx, cancel := context.WithTimeout(ctx, req.Timeout)
			defer cancel()

			service.logger.Debug("Scraping URL with P-tag extraction", "url", url, "index", index)

			content, err := service.ScrapeURL(requestCtx, url)
			mu.Lock()
			defer mu.Unlock()
			if content == nil {
				service.logger.Error("ScrapeURL returned nil content", "url", url)
				result.FailedScrapes = append(result.FailedScrapes, ScrapedContent{
					URL:       url,
					ScrapedAt: time.Now(),
					Success:   false,
					Error:     "Scraper returned nil content",
					Metadata:  make(map[string]string),
					Tags:      []string{},
				})
				result.TotalFailed++
				return
			}
			if err != nil || !content.Success {
				service.logger.Debug("P-tag scraping failed", "url", url, "error", err, "success", content.Success)
				result.FailedScrapes = append(result.FailedScrapes, *content)
				result.TotalFailed++
			} else {
				service.logger.Debug("P-tag scraping successful", "url", url, "content_length", len(content.Content))
				result.SuccessfulScrapes = append(result.SuccessfulScrapes, *content)
				result.TotalSuccessful++
			}
		}(targetURL, i)
	}

	wg.Wait()
	result.Duration = time.Since(startTime)
	successRate := float64(result.TotalSuccessful) / float64(result.TotalRequested) * 100

	service.logger.LogService("scraper", "scrape_multiple_urls_ptag", result.Duration, map[string]interface{}{
		"total_requested": result.TotalRequested,
		"total_failed":    result.TotalFailed,
		"total_success":   result.TotalSuccessful,
		"success_rate":    successRate,
	}, nil)

	return result, nil
}

func (service *ScraperService) ScrapeNewsArticle(ctx context.Context, article *models.NewsArticle) (*models.NewsArticle, error) {
	if article == nil {
		return nil, fmt.Errorf("Article cannot be nil")
	}

	if article.URL == "" {
		return article, fmt.Errorf("Article URL cannot be empty")
	}

	service.logger.Debug("Scraping news article with P-tag extraction", "url", article.URL, "title", article.Title)

	content, err := service.ScrapeURL(ctx, article.URL)
	if err != nil {
		service.logger.Warn("Failed to scrape article", "url", article.URL, "error", err)
		return article, nil
	}

	if content != nil && content.Success {
		if content.Content != "" {
			article.Content = content.Content
			service.logger.Debug("Updated article content with P-tag extraction", "url", article.URL, "length", len(content.Content))
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
	} else {
		service.logger.Debug("P-tag scraping unsuccessful", "url", article.URL, "success", content != nil && content.Success)
	}

	return article, nil
}

func (service *ScraperService) extractTitle(e *colly.HTMLElement) string {
	selectors := []string{
		"article h1", "h1.article-title", "h1.entry-title", "h1.post-title",
		".article-header h1", ".post-header h1", ".entry-header h1",
		"h1", "h2.title", ".title h1", ".title h2",
		"[itemprop='headline']", "[itemprop='title']",
	}

	for _, sel := range selectors {
		if title := e.ChildText(sel); strings.TrimSpace(title) != "" {
			return strings.TrimSpace(title)
		}
	}

	if title := e.ChildText("title"); strings.TrimSpace(title) != "" {
		return strings.TrimSpace(title)
	}

	return ""
}

func (service *ScraperService) extractDescription(e *colly.HTMLElement) string {
	metaSelectors := []string{
		"meta[name='description']", "meta[property='og:description']",
		"meta[name='twitter:description']", "meta[itemprop='description']",
	}

	for _, sel := range metaSelectors {
		if desc := e.ChildAttr(sel, "content"); strings.TrimSpace(desc) != "" {
			return strings.TrimSpace(desc)
		}
	}

	contentSelectors := []string{
		"article p:first-of-type", ".article-summary", ".post-summary",
		".excerpt", ".intro", ".lead", ".summary",
	}

	for _, sel := range contentSelectors {
		if desc := e.ChildText(sel); strings.TrimSpace(desc) != "" {
			return strings.TrimSpace(desc)
		}
	}
	return ""
}

func (service *ScraperService) extractPublishedDate(e *colly.HTMLElement) time.Time {
	if dateStr := e.ChildAttr("[itemprop='datePublished']", "datetime"); dateStr != "" {
		if date, err := time.Parse(time.RFC3339, dateStr); err == nil {
			return date
		}
	}

	if dateStr := e.ChildAttr("time", "datetime"); dateStr != "" {
		if date, err := time.Parse(time.RFC3339, dateStr); err == nil {
			return date
		}
	}

	metaSelectors := []string{
		"meta[property='article:published_time']", "meta[name='publish-date']",
		"meta[name='date']", "meta[name='article:published_time']",
	}

	for _, sel := range metaSelectors {
		if dateStr := e.ChildAttr(sel, "content"); dateStr != "" {
			formats := []string{
				time.RFC3339,
				"2006-01-02T15:04:05Z",
				"2006-01-02 15:04:05",
				"2006-01-02",
			}
			for _, f := range formats {
				if date, err := time.Parse(f, dateStr); err == nil {
					return date
				}
			}
		}
	}

	return time.Time{}
}

func (service *ScraperService) extractMainImage(e *colly.HTMLElement) string {
	metaSelectors := []string{
		"meta[property='og:image']", "meta[name='twitter:image']", "meta[itemprop='image']",
	}

	for _, sel := range metaSelectors {
		if img := e.ChildAttr(sel, "content"); strings.TrimSpace(img) != "" {
			return strings.TrimSpace(img)
		}
	}

	imgSelectors := []string{
		"article img:first-of-type", ".article-image img", ".featured-image img",
		".post-image img", ".hero-image img", ".lead-image img",
	}

	for _, sel := range imgSelectors {
		if img := e.ChildAttr(sel, "src"); strings.TrimSpace(img) != "" {
			return strings.TrimSpace(img)
		}
	}
	return ""
}

func (service *ScraperService) extractTags(e *colly.HTMLElement) []string {
	var tags []string
	if keywords := e.ChildAttr("meta[name='keywords']", "content"); strings.TrimSpace(keywords) != "" {
		for _, tag := range strings.Split(keywords, ",") {
			trimmed := strings.ToLower(strings.TrimSpace(tag))
			if trimmed != "" {
				tags = append(tags, trimmed)
			}
		}
	}

	tagSelectors := []string{
		".tags a", ".article-tags a", ".post-tags a", "[rel='tag']",
		".categories a", ".taxonomy a",
	}

	for _, sel := range tagSelectors {
		e.ForEach(sel, func(_ int, el *colly.HTMLElement) {
			tagText := strings.TrimSpace(el.Text)
			if tagText != "" {
				tags = append(tags, strings.ToLower(tagText))
			}
		})
	}
	return service.cleanTags(tags)
}

func (service *ScraperService) extractAuthor(e *colly.HTMLElement) string {
	selectors := []string{
		"[rel='author']", "[itemprop='author'] [itemprop='name']",
		"[itemprop='author']", ".author-name", ".article-author",
		".byline-author", ".post-author", ".author", ".by-author", ".byline",
	}

	for _, sel := range selectors {
		if author := e.ChildText(sel); strings.TrimSpace(author) != "" {
			return strings.TrimSpace(author)
		}
	}

	if author := e.ChildAttr("meta[name='author']", "content"); strings.TrimSpace(author) != "" {
		return strings.TrimSpace(author)
	}
	return ""
}

func (service *ScraperService) extractCharset(e *colly.HTMLElement) string {
	if charset := e.Attr("meta[charset]"); charset != "" {
		return charset
	}
	if contentType := e.Attr("meta[http-equiv='Content-Type'] content"); contentType != "" {
		if strings.Contains(contentType, "charset=") {
			parts := strings.Split(contentType, "charset=")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "utf-8"
}

func (service *ScraperService) cleanTags(tags []string) []string {
	seen := make(map[string]bool)
	var cleaned []string

	for _, tag := range tags {
		tag = strings.TrimSpace(tag)
		tag = strings.ToLower(tag)
		if tag != "" && !seen[tag] && len(tag) > 1 && len(tag) < 50 {
			seen[tag] = true
			cleaned = append(cleaned, tag)
		}
	}

	if len(cleaned) > 10 {
		cleaned = cleaned[:10]
	}
	return cleaned
}

// setupCallbacks sets up enhanced request callbacks
func (service *ScraperService) setupCallbacks() {
	service.collector.OnRequest(func(r *colly.Request) {
		service.mu.Lock()
		userAgent := service.userAgents[service.uaIndex]
		service.uaIndex = (service.uaIndex + 1) % len(service.userAgents)
		service.mu.Unlock()

		r.Headers.Set("User-Agent", userAgent)
		r.Headers.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
		r.Headers.Set("Accept-Language", "en-US,en;q=0.5")
		r.Headers.Set("Accept-Encoding", "gzip, deflate, br")
		r.Headers.Set("DNT", "1")
		r.Headers.Set("Connection", "keep-alive")
		r.Headers.Set("Upgrade-Insecure-Requests", "1")
		r.Headers.Set("Sec-Fetch-Dest", "document")
		r.Headers.Set("Sec-Fetch-Mode", "navigate")
		r.Headers.Set("Sec-Fetch-Site", "none")
		r.Headers.Set("Cache-Control", "max-age=0")
	})
}

func (service *ScraperService) HealthCheck(ctx context.Context) error {
	testURL := "https://httpbin.org/html"

	testCtx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	content, err := service.ScrapeURL(testCtx, testURL)
	if err != nil {
		return fmt.Errorf("P-tag health check scrape failed: %w", err)
	}

	if content == nil || !content.Success {
		return fmt.Errorf("P-tag health check scrape was unsuccessful")
	}
	return nil
}

// Helper to safely truncate string for logging
func safeTruncate(s string, length int) string {
	if len(s) <= length {
		return s
	}
	return s[:length] + "..."
}

// Helper min function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
