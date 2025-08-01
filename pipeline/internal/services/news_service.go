package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type NewsService struct {
	client *http.Client
	apiKey string
	logger *logger.Logger
	config config.EtcConfig
}

type NewsAPIResponse struct {
	Status       string        `json:"status"`
	TotalResults int           `json:"total_results"`
	Articles     []APIArticles `json:"articles"`
	Code         string        `json:"code,omitempty"`
	Message      string        `json:"message,omitempty"`
}

type APIArticles struct {
	Source      APISource `json:"source"`
	Author      string    `json:"author"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	URL         string    `json:"url"`
	URLToImage  string    `json:"urlToImage"`
	PublishedAt string    `json:"publishedAt"`
	Content     string    `json:"content"`
}

type APISource struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type SearchRequest struct {
	Query          string
	Keywords       []string
	Language       string
	SortBy         string
	Sources        []string
	Domains        []string
	ExcludeDomains []string
	From           *time.Time
	To             *time.Time
	PageSize       int
	Page           int
}

type HeadlinesRequest struct {
	Country  string
	Category string
	Sources  []string
	PageSize int
	Page     int
}

const (
	NewsAPIBaseURL  = "https://newsapi.org/v2"
	MaxPageSize     = 100
	DefaultPageSize = 100
)

func NewNewsService(config config.EtcConfig, logger *logger.Logger) (*NewsService, error) {
	if config.NewsApiKey == "" {
		return nil, fmt.Errorf("NewsApiKey is required")
	}

	client := &http.Client{
		Timeout: time.Second * 30,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			MaxIdleConnsPerHost: 5,
			DisableCompression:  false,
			IdleConnTimeout:     time.Second * 30,
		},
	}

	service := &NewsService{
		client: client,
		apiKey: config.NewsApiKey,
		logger: logger,
		config: config,
	}

	if err := service.TestConnection(); err != nil {
		return nil, fmt.Errorf("NewsAPI connection test failed: %w", err)
	}

	logger.Info("NewsService successfully created and Initialized", "base_url", NewsAPIBaseURL, "rate_limit", "1000 requests/day (free tier)")

	return service, nil
}

func (service *NewsService) TestConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*1000)
	defer cancel()

	_, err := service.GetTopHeadlines(ctx, &HeadlinesRequest{
		Country:  "us",
		PageSize: 1,
	})

	if err != nil {
		return fmt.Errorf("NewsAPI test connection failed: %w", err)
	}

	service.logger.Info("NewsAPI test connection succeeded")
	return nil

}

func (service *NewsService) SearchEverything(ctx context.Context, req *SearchRequest) ([]models.NewsArticle, error) {
	if req == nil {
		return nil, fmt.Errorf("SearchRequest is required")
	}

	startTime := time.Now()

	service.logger.LogService("news_api", "search_everything", 0, map[string]interface{}{
		"query":     req.Query,
		"keywords":  req.Keywords,
		"page_size": 100,
		"page":      req.Page,
		"sort_by":   req.SortBy,
	}, nil)

	searchQuery := service.buildSearchQuery(req)
	if searchQuery == "" {
		return nil, fmt.Errorf("Search Query cannot be empty")
	}

	params := service.buildEverythingParams(req, searchQuery)

	articles, err := service.makeAPIRequest(ctx, "everything", params)
	if err != nil {
		service.logger.LogService("news_api", "search_everything", time.Since(startTime), map[string]interface{}{
			"query": req.Query,
		}, nil)
		return nil, fmt.Errorf("NewsAPI search everything failed: %w", err)
	}

	result := service.convertToDesiredFormat(articles)
	service.logger.Info(result)
	service.logger.LogService("news_api", "search_everything", time.Since(startTime), map[string]interface{}{
		"query":          req.Query,
		"articles_found": len(result),
		"articles":       articles,
		"total_results":  len(result),
	}, nil)

	fmt.Println("news agent found : ")
	fmt.Println(result)
	fmt.Println()

	return result, nil
}

func (service *NewsService) buildEverythingParams(req *SearchRequest, searchQuery string) url.Values {
	params := url.Values{}
	params.Add("q", searchQuery)
	params.Set("apiKey", service.apiKey)
	if req.Language == "" {
		params.Set("language", "en")
	} else {
		params.Set("language", req.Language)
	}

	// Set sort order
	if req.SortBy == "" {
		params.Set("sortBy", "relevancy")
	} else {
		params.Set("sortBy", req.SortBy)
	}

	if req.PageSize == 0 {
		params.Set("pageSize", strconv.Itoa(DefaultPageSize))
	} else {
		if req.PageSize > MaxPageSize {
			req.PageSize = MaxPageSize
		}
		params.Set("pageSize", strconv.Itoa(req.PageSize))
	}
	if req.Page == 0 {
		params.Set("page", "1")
	} else {
		params.Set("page", strconv.Itoa(req.Page))
	}

	// Add date filters
	if req.From != nil {
		params.Set("from", req.From.Format("2006-01-02"))
	}
	if req.To != nil {
		params.Set("to", req.To.Format("2006-01-02"))
	}

	// Add sources filter
	if len(req.Sources) > 0 {
		params.Set("sources", strings.Join(req.Sources, ","))
	}

	// Add domains filter
	if len(req.Domains) > 0 {
		params.Set("domains", strings.Join(req.Domains, ","))
	}

	// Add exclude domains filter
	if len(req.ExcludeDomains) > 0 {
		params.Set("excludeDomains", strings.Join(req.ExcludeDomains, ","))
	}

	return params

}

func (service *NewsService) buildSearchQuery(req *SearchRequest) string {
	var queryParts []string

	if req.Query != "" {
		queryParts = append(queryParts, req.Query)
	}

	return strings.Join(queryParts, " AND ")

}

func (service *NewsService) GetTopHeadlines(ctx context.Context, request *HeadlinesRequest) ([]models.NewsArticle, error) {
	if request == nil {
		request = &HeadlinesRequest{}
	}
	startTime := time.Now()

	service.logger.LogService("news_api", "get_top_headlines", 0, map[string]interface{}{
		"Country":   request.Country,
		"Category":  request.Category,
		"page_size": request.PageSize,
		"page":      request.Page,
	}, nil)

	params := service.BuildHeadlinesParams(request)

	articles, err := service.makeAPIRequest(ctx, "top-headlines", params)
	if err != nil {
		service.logger.LogService("news_api", "get_top_headlines", time.Since(startTime), map[string]interface{}{
			"country":  request.Country,
			"category": request.Category,
		}, nil)
		return nil, fmt.Errorf("news api get top headlines: %w", err)
	}

	result := service.convertToDesiredFormat(articles)

	service.logger.LogService("news_api", "get_top_headlines", time.Since(startTime), map[string]interface{}{
		"country":        request.Country,
		"category":       request.Category,
		"articles_found": result,
	}, nil)

	return result, nil
}

func (service *NewsService) SearchByKeywords(ctx context.Context, keywords []string, maxResults int) ([]models.NewsArticle, error) {
	if len(keywords) == 0 {
		return nil, fmt.Errorf("Keywords cannot be empty")
	}

	req := &SearchRequest{
		Keywords: keywords,
		Query:    strings.Join(keywords, " OR "),
		PageSize: maxResults,
		Page:     1,
		SortBy:   "popularity",
		Language: "en",
	}

	return service.SearchEverything(ctx, req)

}

func (service *NewsService) SearchRecentNews(ctx context.Context, query string, hoursBack int, maxResults int) ([]models.NewsArticle, error) {
	if len(query) == 0 {
		return nil, fmt.Errorf("Query cannot be empty")
	}

	from := time.Now().Add(-time.Duration(hoursBack) * time.Hour)

	req := &SearchRequest{
		Query:    query,
		From:     &from,
		PageSize: MaxPageSize,
		Page:     1,
		SortBy:   "publishedAt",
		Language: "en",
	}

	return service.SearchEverything(ctx, req)

}

func (service *NewsService) HealthCheck(ctx context.Context) error {
	testCtx, cancel := context.WithTimeout(ctx, 1000*time.Second)
	defer cancel()

	_, err := service.GetTopHeadlines(testCtx, &HeadlinesRequest{
		Country:  "us",
		PageSize: 5,
	})

	if err != nil {
		return fmt.Errorf("news api health check failed : %w", err)
	}

	return nil
}

func (service *NewsService) makeAPIRequest(ctx context.Context, endpoint string, params url.Values) ([]APIArticles, error) {
	fullURL := fmt.Sprintf("%s/%s?%s", NewsAPIBaseURL, endpoint, params.Encode())

	req, err := http.NewRequestWithContext(ctx, "GET", fullURL, nil)
	if err != nil {
		return nil, fmt.Errorf("news api request creation failed: %w", err)
	}

	req.Header.Set("User-Agent", "Anya-AI-News-Assistant/1.0")

	resp, err := service.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("news api request execution failed: %w", err)
	}
	defer resp.Body.Close()

	var apiResponse NewsAPIResponse
	err = json.NewDecoder(resp.Body).Decode(&apiResponse)
	if err != nil {
		return nil, fmt.Errorf("news api response decoding failed: %w", err)
	}

	if apiResponse.Status != "ok" {
		return nil, fmt.Errorf("news api error: %s - %s", apiResponse.Code, apiResponse.Message)
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, models.NewRateLimitError("NEWSAPI_RATE_LIMIT", "NewsAPI rate limit Exceeded", 60*time.Minute)
	}

	return apiResponse.Articles, nil

}

func (service *NewsService) convertToDesiredFormat(apiArticles []APIArticles) []models.NewsArticle {
	articles := make([]models.NewsArticle, 0, len(apiArticles))

	for _, apiArticle := range apiArticles {
		// Skip invalid articles
		if apiArticle.Title == "" || apiArticle.URL == "" {
			continue
		}

		// Parse published date with fallback
		publishedAt, err := time.Parse("2006-01-02T15:04:05Z", apiArticle.PublishedAt)
		if err != nil {
			// Try alternative format
			publishedAt, err = time.Parse("2006-01-02T15:04:05.000Z", apiArticle.PublishedAt)
			if err != nil {
				publishedAt = time.Now() // Fallback to current time
			}
		}

		articleID := service.generateArticleID(apiArticle.URL)

		article := models.NewsArticle{
			ID:          articleID,
			Title:       apiArticle.Title,
			URL:         apiArticle.URL,
			Source:      apiArticle.Source.Name,
			PublishedAt: publishedAt,
			Description: apiArticle.Description,
			Content:     apiArticle.Content,
			ImageURL:    apiArticle.URLToImage,
			Author:      apiArticle.Author,
		}

		articles = append(articles, article)
	}

	return articles
}

func (service *NewsService) generateArticleID(url string) string {
	return fmt.Sprintf("news_%d", service.simpleHash(url))
}

func (service *NewsService) simpleHash(url string) uint32 {
	h := uint32(0)
	for _, c := range url {
		h = h*31 + uint32(c)
	}
	return h
}

func (service *NewsService) BuildHeadlinesParams(request *HeadlinesRequest) url.Values {
	params := url.Values{}
	params.Set("apiKey", service.apiKey)

	if request.Country != "" {
		params.Set("country", request.Country)
	}
	if request.Category != "" {
		params.Set("category", request.Category)
	}
	if len(request.Sources) > 0 {
		params.Set("sources", strings.Join(request.Sources, ","))
	}

	if request.PageSize == 0 {
		params.Set("pageSize", strconv.Itoa(DefaultPageSize))
	} else if request.PageSize <= MaxPageSize {
		if request.PageSize > MaxPageSize {
			params.Set("pageSize", strconv.Itoa(MaxPageSize))
		} else {
			params.Set("pageSize", strconv.Itoa(request.PageSize))
		}
	}

	if request.Page == 0 {
		params.Set("page", "1")
	} else {
		params.Set("page", strconv.Itoa(request.Page))
	}

	return params

}
