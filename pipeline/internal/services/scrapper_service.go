package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
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

type ScoredElement struct {
	Selection *goquery.Selection
	Score     int
	Text      string
}

func NewScraperService(config config.ScraperConfig, logger *logger.Logger) (*ScraperService, error) {
	collector := colly.NewCollector(
		colly.Debugger(&debug.LogDebugger{}),
		colly.UserAgent("Anya-AI-News-Assistant/1.0 (+https://anya-ai.com/bot)"),
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
	logger.Info("Enhanced Scraper Service initialized successfully",
		"rate_limit", "5 concurrent requests",
		"delay", "3 seconds between requests",
		"timeout", "60 seconds",
		"content_extraction", "readability-enhanced")

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

	service.logger.Info("Starting enhanced scrape",
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

		service.logger.Debug("Enhanced scraper request sent",
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
		service.logger.Debug("Processing HTML content with enhanced extraction",
			"url", targetURL,
			"html_length", len(e.Text))

		// Enhanced content extraction
		content.Content = service.extractArticleContent(e)
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

		service.logger.Info("Enhanced content extraction results",
			"url", targetURL,
			"has_title", hasTitle,
			"has_content", hasContent,
			"has_description", hasDescription,
			"title", safeTruncate(content.Title, 50),
			"content_length", len(content.Content),
			"quality_score", service.calculateContentQuality(content),
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
		service.logger.Debug("Enhanced scraping completed", "url", targetURL)
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
	service.logger.LogService("scraper", "scraper_url_enhanced", duration, map[string]interface{}{
		"url":            targetURL,
		"success":        content.Success,
		"content_length": len(content.Content),
		"title":          content.Title != "",
		"description":    content.Description != "",
		"status_code":    httpStatusCode,
		"response_size":  responseSize,
		"quality_score":  service.calculateContentQuality(content),
		"error":          content.Error,
	}, scrapingError)

	return content, nil
}

func (service *ScraperService) ScrapeMultipleURLs(ctx context.Context, req *ScrapingRequest) (*ScrapingResult, error) {
	if req == nil {
		return nil, fmt.Errorf("Scraping request cannot be nil")
	}
	if len(req.URLs) == 0 {
		return nil, fmt.Errorf("Scraping Request URLs cannot be empty")
	}

	startTime := time.Now()
	service.logger.Info("Starting enhanced multiple URL scraping",
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

			service.logger.Debug("Scraping URL with enhanced extraction", "url", url, "index", index)

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
				service.logger.Debug("Enhanced scraping failed", "url", url, "error", err, "success", content.Success)
				result.FailedScrapes = append(result.FailedScrapes, *content)
				result.TotalFailed++
			} else {
				service.logger.Debug("Enhanced scraping successful", "url", url, "content_length", len(content.Content))
				result.SuccessfulScrapes = append(result.SuccessfulScrapes, *content)
				result.TotalSuccessful++
			}
		}(targetURL, i)
	}

	wg.Wait()
	result.Duration = time.Since(startTime)
	successRate := float64(result.TotalSuccessful) / float64(result.TotalRequested) * 100

	service.logger.LogService("scraper", "scrape_multiple_urls_enhanced", result.Duration, map[string]interface{}{
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

	service.logger.Debug("Scraping news article with enhanced extraction", "url", article.URL, "title", article.Title)

	content, err := service.ScrapeURL(ctx, article.URL)
	if err != nil {
		service.logger.Warn("Failed to scrape article", "url", article.URL, "error", err)
		return article, nil
	}

	if content != nil && content.Success {
		if content.Content != "" {
			article.Content = content.Content
			service.logger.Debug("Updated article content with enhanced extraction", "url", article.URL, "length", len(content.Content))
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
		service.logger.Debug("Enhanced scraping unsuccessful", "url", article.URL, "success", content != nil && content.Success)
	}

	return article, nil
}

// ================ ENHANCED CONTENT EXTRACTION METHODS ================

// extractArticleContent uses readability-inspired heuristics to extract main article content
func (service *ScraperService) extractArticleContent(e *colly.HTMLElement) string {
	// First, remove known noise elements before processing
	service.removeNoiseElements(e)

	// Try structured content containers first
	content := service.extractFromStructuredContainers(e)

	if content == "" {
		// Fallback to heuristic-based extraction
		content = service.extractUsingHeuristics(e)
	}

	if content == "" {
		// Final fallback to paragraph extraction
		content = service.extractFromParagraphs(e)
	}

	// Clean and validate the extracted content
	content = service.cleanContent(content)

	// Validate minimum content quality
	if len(strings.TrimSpace(content)) < 100 {
		service.logger.Debug("Extracted content too short, trying fallback", "length", len(content))
		return service.extractFallbackContent(e)
	}

	// Limit maximum content length
	if len(content) > 12000 {
		content = content[:12000] + "..."
	}

	service.logger.Debug("Successfully extracted article content",
		"length", len(content),
		"preview", safeTruncate(content, 100))

	return content
}

// removeNoiseElements removes common noise elements before content extraction
func (service *ScraperService) removeNoiseElements(e *colly.HTMLElement) {
	noiseSelectors := []string{
		"script", "style", "noscript", "iframe",
		"nav", "header", "footer", "aside",
		".advertisement", ".ads", ".ad", ".banner",
		".sidebar", ".social", ".share", ".comments",
		".newsletter", ".subscribe", ".popup", ".modal",
		".navigation", ".menu", ".related", ".recommended",
		".author-bio", ".tags", ".categories",
		"[id*='ad']", "[class*='ad']", "[id*='social']",
		"[class*='share']", "[class*='comment']",
	}

	for _, selector := range noiseSelectors {
		e.DOM.Find(selector).Remove()
	}

	service.logger.Debug("Removed noise elements from DOM")
}

// extractFromStructuredContainers tries to extract from semantic HTML containers
func (service *ScraperService) extractFromStructuredContainers(e *colly.HTMLElement) string {
	// Priority order for content containers
	containerSelectors := []string{
		"article",
		"main",
		"[role='main']",
		".main-content",
		".article-content",
		".post-content",
		".entry-content",
		".content",
		"#main",
		"#content",
	}

	for _, selector := range containerSelectors {
		if container := e.DOM.Find(selector).First(); container.Length() > 0 {
			content := service.extractTextFromContainer(container)
			if len(strings.TrimSpace(content)) > 200 {
				service.logger.Debug("Extracted content from structured container",
					"selector", selector, "length", len(content))
				return content
			}
		}
	}

	return ""
}

// extractTextFromContainer extracts clean text from a specific container
func (service *ScraperService) extractTextFromContainer(container *goquery.Selection) string {
	var paragraphs []string

	// Extract paragraphs with content filtering
	container.Find("p").Each(func(i int, s *goquery.Selection) {
		text := strings.TrimSpace(s.Text())

		// Filter out short paragraphs and noise
		if len(text) > 50 && !service.isNoiseText(text) {
			paragraphs = append(paragraphs, text)
		}
	})

	// If no good paragraphs, try other text elements
	if len(paragraphs) == 0 {
		container.Find("div, section, span").Each(func(i int, s *goquery.Selection) {
			text := strings.TrimSpace(s.Text())

			// Only include elements with substantial text and low link density
			if len(text) > 100 && service.calculateLinkDensity(s) < 0.3 {
				paragraphs = append(paragraphs, text)
			}
		})
	}

	return strings.Join(paragraphs, "\n\n")
}

// extractUsingHeuristics uses content scoring to find the main article
func (service *ScraperService) extractUsingHeuristics(e *colly.HTMLElement) string {
	var candidates []ScoredElement

	// Score potential content containers
	e.DOM.Find("div, section, article").Each(func(i int, s *goquery.Selection) {
		text := strings.TrimSpace(s.Text())

		if len(text) < 200 {
			return // Skip short content
		}

		score := service.scoreElement(s, text)

		if score > 50 { // Minimum threshold
			candidates = append(candidates, ScoredElement{
				Selection: s,
				Score:     score,
				Text:      text,
			})
		}
	})

	// Return the highest scored content
	if len(candidates) > 0 {
		// Sort by score descending
		bestCandidate := candidates[0]
		for _, candidate := range candidates[1:] {
			if candidate.Score > bestCandidate.Score {
				bestCandidate = candidate
			}
		}

		service.logger.Debug("Selected content using heuristics",
			"score", bestCandidate.Score, "length", len(bestCandidate.Text))

		return bestCandidate.Text
	}

	return ""
}

// scoreElement scores an element based on content quality indicators
func (service *ScraperService) scoreElement(s *goquery.Selection, text string) int {
	score := 0

	// Base score from text length
	score += len(text) / 100

	// Bonus for paragraph count
	pCount := s.Find("p").Length()
	score += pCount * 10

	// Penalty for high link density
	linkDensity := service.calculateLinkDensity(s)
	if linkDensity > 0.5 {
		score -= 50
	} else if linkDensity < 0.1 {
		score += 20
	}

	// Bonus for typical content class names
	class := s.AttrOr("class", "")
	id := s.AttrOr("id", "")

	contentIndicators := []string{"content", "article", "post", "main", "body", "entry"}
	for _, indicator := range contentIndicators {
		if strings.Contains(class, indicator) || strings.Contains(id, indicator) {
			score += 25
			break
		}
	}

	// Penalty for noise class names
	noiseIndicators := []string{"sidebar", "nav", "menu", "ad", "comment", "footer", "header"}
	for _, indicator := range noiseIndicators {
		if strings.Contains(class, indicator) || strings.Contains(id, indicator) {
			score -= 30
			break
		}
	}

	return score
}

// calculateLinkDensity calculates the ratio of link text to total text
func (service *ScraperService) calculateLinkDensity(s *goquery.Selection) float64 {
	totalText := len(strings.TrimSpace(s.Text()))
	if totalText == 0 {
		return 1.0
	}

	linkText := 0
	s.Find("a").Each(func(i int, link *goquery.Selection) {
		linkText += len(strings.TrimSpace(link.Text()))
	})

	return float64(linkText) / float64(totalText)
}

// extractFromParagraphs fallback method to extract from paragraphs
func (service *ScraperService) extractFromParagraphs(e *colly.HTMLElement) string {
	var paragraphs []string

	e.DOM.Find("p").Each(func(i int, s *goquery.Selection) {
		text := strings.TrimSpace(s.Text())

		if len(text) > 80 && !service.isNoiseText(text) {
			paragraphs = append(paragraphs, text)
		}
	})

	return strings.Join(paragraphs, "\n\n")
}

// extractFallbackContent final fallback for difficult pages
func (service *ScraperService) extractFallbackContent(e *colly.HTMLElement) string {
	// Try to find any substantial text blocks
	var textBlocks []string

	e.DOM.Find("*").Each(func(i int, s *goquery.Selection) {
		// Skip if it has child elements (not a leaf text node)
		if s.Children().Length() > 2 {
			return
		}

		text := strings.TrimSpace(s.Text())
		if len(text) > 150 && !service.isNoiseText(text) {
			textBlocks = append(textBlocks, text)
		}
	})

	if len(textBlocks) > 0 {
		return strings.Join(textBlocks[:min(3, len(textBlocks))], "\n\n")
	}

	return ""
}

// isNoiseText checks if text contains common noise patterns
func (service *ScraperService) isNoiseText(text string) bool {
	lowerText := strings.ToLower(text)

	noisePatterns := []string{
		"subscribe", "newsletter", "advertisement", "cookie",
		"privacy policy", "terms of service", "follow us",
		"share this", "related articles", "read more",
		"click here", "sign up", "log in", "contact us",
	}

	for _, pattern := range noisePatterns {
		if strings.Contains(lowerText, pattern) {
			return true
		}
	}

	// Check for excessive capitalization (likely headers/navigation)
	upperCount := 0
	for _, r := range text {
		if r >= 'A' && r <= 'Z' {
			upperCount++
		}
	}

	if len(text) > 0 && float64(upperCount)/float64(len(text)) > 0.7 {
		return true
	}

	return false
}

// ================ ENHANCED CONTENT CLEANING METHODS ================

// cleanContent performs aggressive content cleaning
func (service *ScraperService) cleanContent(content string) string {
	if content == "" {
		return content
	}

	// Step 1: Normalize whitespace
	content = service.normalizeWhitespace(content)

	// Step 2: Remove noise patterns
	content = service.removeNoisePatterns(content)

	// Step 3: Clean formatting
	content = service.cleanFormatting(content)

	// Step 4: Remove repeated content
	content = service.removeRepeatedContent(content)

	// Step 5: Final cleanup
	content = strings.TrimSpace(content)

	return content
}

// normalizeWhitespace fixes spacing issues
func (service *ScraperService) normalizeWhitespace(content string) string {
	// Replace multiple spaces with single space
	re := regexp.MustCompile(`\s+`)
	content = re.ReplaceAllString(content, " ")

	// Fix paragraph spacing
	re = regexp.MustCompile(`\n\s*\n`)
	content = re.ReplaceAllString(content, "\n\n")

	// Remove trailing/leading whitespace per line
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimSpace(line)
	}

	return strings.Join(lines, "\n")
}

// removeNoisePatterns removes common noise text
func (service *ScraperService) removeNoisePatterns(content string) string {
	// Comprehensive noise patterns
	patterns := []string{
		`(?i)\s*javascript:void\(0\)\s*`,
		`(?i)\s*advertisement\s*`,
		`(?i)\s*sponsored content\s*`,
		`(?i)\s*subscribe to our newsletter\s*`,
		`(?i)\s*follow us on social media\s*`,
		`(?i)\s*share this article\s*`,
		`(?i)\s*read more articles?\s*`,
		`(?i)\s*click here for more\s*`,
		`(?i)\s*contact us for\s*`,
		`(?i)\s*privacy policy\s*`,
		`(?i)\s*terms of service\s*`,
		`(?i)\s*cookie policy\s*`,
		`(?i)\s*all rights reserved\s*`,
		`(?i)\s*copyright \d+\s*`,
		`(?i)\s*powered by\s*`,
		`(?i)\s*related articles?\s*`,
		`(?i)\s*you might also like\s*`,
		`(?i)\s*trending now\s*`,
		`(?i)\s*most popular\s*`,
		`(?i)\s*recommended for you\s*`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		content = re.ReplaceAllString(content, "")
	}

	return content
}

// cleanFormatting removes formatting artifacts
func (service *ScraperService) cleanFormatting(content string) string {
	// Remove excessive punctuation
	re := regexp.MustCompile(`[.]{3,}`)
	content = re.ReplaceAllString(content, "...")

	// Remove excessive dashes
	re = regexp.MustCompile(`[-]{3,}`)
	content = re.ReplaceAllString(content, "---")

	// Remove stray formatting characters
	re = regexp.MustCompile(`[_*]{2,}`)
	content = re.ReplaceAllString(content, "")

	// Clean up quotes
	content = strings.ReplaceAll(content, `"`, `\"`)
	content = strings.ReplaceAll(content, `"`, `\"`)
	content = strings.ReplaceAll(content, "'", "'")
	content = strings.ReplaceAll(content, "'", "'")

	return content
}

// removeRepeatedContent removes duplicate sentences/paragraphs
func (service *ScraperService) removeRepeatedContent(content string) string {
	paragraphs := strings.Split(content, "\n\n")
	seen := make(map[string]bool)
	var unique []string

	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		if paragraph == "" {
			continue
		}

		// Create a normalized version for comparison
		normalized := strings.ToLower(strings.TrimSpace(paragraph))

		if !seen[normalized] && len(paragraph) > 20 {
			seen[normalized] = true
			unique = append(unique, paragraph)
		}
	}

	return strings.Join(unique, "\n\n")
}

// ================ ORIGINAL EXTRACTION HELPERS ================

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

// ================ UTILITY METHODS ================

func (service *ScraperService) calculateContentQuality(content *ScrapedContent) int {
	score := 0

	if content.Title != "" {
		score += 20
	}
	if len(content.Content) > 500 {
		score += 40
	}
	if content.Description != "" {
		score += 20
	}
	if content.Author != "" {
		score += 10
	}
	if !content.PublishedAt.IsZero() {
		score += 10
	}

	return score
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
		return fmt.Errorf("enhanced health check scrape failed: %w", err)
	}

	if content == nil || !content.Success {
		return fmt.Errorf("enhanced health check scrape was unsuccessful")
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
