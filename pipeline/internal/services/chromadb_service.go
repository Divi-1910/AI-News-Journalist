package services

import (
	"anya-ai-pipeline/internal/config"
	"anya-ai-pipeline/internal/models"
	"anya-ai-pipeline/internal/pkg/logger"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

type ChromaDBService struct {
	client   *http.Client
	baseURL  *url.URL
	logger   *logger.Logger
	tenant   string
	database string
}

type Collection struct {
	ID       string            `json:"id"`
	Content  string            `json:"document"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type Document struct {
	ID        string                 `json:"id"`
	Content   string                 `json:"document"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Embedding []float64              `json:"embedding,omitempty"`
}

type QueryRequest struct {
	QueryEmbeddings [][]float64            `json:"query_embeddings"`
	NResults        int                    `json:"n_results"`
	Where           map[string]interface{} `json:"where,omitempty"`
	Include         []string               `json:"include,omitempty"`
}

type QueryResponse struct {
	IDs        [][]string                 `json:"ids"`
	Documents  [][]string                 `json:"documents"`
	Metadatas  [][]map[string]interface{} `json:"metadatas"`
	Distances  [][]float64                `json:"distances"`
	Embeddings [][][]float64              `json:"embeddings"`
	Include    []string                   `json:"include"`
}

type AddRequest struct {
	Documents  []string                 `json:"documents,omitempty"`
	Metadatas  []map[string]interface{} `json:"metadatas,omitempty"`
	IDs        []string                 `json:"ids"`
	Embeddings [][]float64              `json:"embeddings"`
}

type SearchResult struct {
	Document   models.NewsArticle `json:"document"`
	Similarity float64            `json:"similarity"`
	Distance   float64            `json:"distance"`
}

const (
	NewsCollectionName = "news_articles"
	DefaultTopK        = 8
	DefaultTenant      = "default_tenant"
	DefaultDatabase    = "default_database"
)

func NewChromaDBService(config config.EtcConfig, log *logger.Logger) (*ChromaDBService, error) {
	if config.ChromaDBURL == "" {
		return nil, fmt.Errorf("ChromaDB URL is required")
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			IdleConnTimeout:     30 * time.Second,
			DisableCompression:  false,
			MaxIdleConnsPerHost: 10,
		},
	}

	baseURL, err := url.Parse(config.ChromaDBURL)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse ChromaDB URL: %w", err)
	}

	service := &ChromaDBService{
		client:   client,
		baseURL:  baseURL,
		logger:   log,
		tenant:   DefaultTenant,
		database: DefaultDatabase,
	}

	if err := service.initialize(); err != nil {
		return nil, fmt.Errorf("Failed to initialize ChromaDB service: %w", err)
	}

	log.Info("ChromaDB service initialized successfully", "base_url", config.ChromaDBURL,
		"collection", NewsCollectionName)

	return service, nil

}

func (service *ChromaDBService) initialize() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := service.HealthCheck(ctx); err != nil {
		return fmt.Errorf("Failed to connect to ChromaDB service: %w", err)
	}

	if err := service.createOrGetCollection(ctx, NewsCollectionName); err != nil {
		return fmt.Errorf("Collection setup failed : %w", err)
	}

	service.logger.Info("ChromaDB service initialized successfully")
	return nil
}

func (service *ChromaDBService) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/api/v2/healthcheck", service.baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("Failed to create chromadb healthcheck request: %w", err)
	}

	resp, err := service.client.Do(req)
	if err != nil {
		return fmt.Errorf("Failed chromadb healthcheck request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Failed chromadb healthcheck request: invalid status code: %d", resp.StatusCode)
	}

	return nil

}

func (service *ChromaDBService) createOrGetCollection(ctx context.Context, collectionName string) error {
	collection, err := service.getCollection(ctx, collectionName)
	if err == nil && collection != nil {
		service.logger.Info("Using Exisiting Collection", "collection", collectionName)
		return nil
	}

	newCollection := Collection{
		ID: collectionName,
		Metadata: map[string]string{
			"description": "News Articles for semantic search",
			"created_at":  time.Now().Format(time.RFC3339),
		},
	}

	if err := service.createCollection(ctx, newCollection); err != nil {
		return fmt.Errorf("Failed to create new collection: %w", err)
	}

	service.logger.Info("Created new collection", "collection", newCollection)
	return nil

}

func (service *ChromaDBService) createCollection(ctx context.Context, collectionName Collection) error {
	url := fmt.Sprintf("%s/api/v2/tenants/%s/databases/%s/collections", service.baseURL, service.tenant, service.database)

	payload := map[string]interface{}{
		"name":         collectionName.ID,
		"metadata":     collectionName.Metadata,
		"get_or_create": true,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("Failed to marshall collection: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("Failed to create new collection request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := service.client.Do(req)

	if err != nil {
		return fmt.Errorf("Failed new collection request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("Failed new collection request: invalid status code: %d", resp.StatusCode)
	}
	return nil

}

func (service *ChromaDBService) getCollection(ctx context.Context, collectionName string) (*Collection, error) {
	// First get collections list to find the collection ID
	url := fmt.Sprintf("%s/api/v2/tenants/%s/databases/%s/collections", service.baseURL, service.tenant, service.database)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("Failed to create chromadb get collections request: %w", err)
	}
	resp, err := service.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Failed chromadb get collections request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Failed chromadb get collections request: invalid status code: %d", resp.StatusCode)
	}

	var collections []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&collections); err != nil {
		return nil, fmt.Errorf("Failed to decode collections response: %w", err)
	}

	// Find collection by name
	for _, col := range collections {
		if name, ok := col["name"].(string); ok && name == collectionName {
			collection := &Collection{
				ID: name,
			}
			if metadata, ok := col["metadata"].(map[string]interface{}); ok {
				collection.Metadata = make(map[string]string)
				for k, v := range metadata {
					if str, ok := v.(string); ok {
						collection.Metadata[k] = str
					}
				}
			}
			return collection, nil
		}
	}

	return nil, fmt.Errorf("Collection %s not found", collectionName)
}

func (service *ChromaDBService) StoreArticles(ctx context.Context, articles []models.NewsArticle, embeddings [][]float64) error {
	if len(articles) != len(embeddings) {
		return fmt.Errorf("mismatch between articles (%d) and embeddings (%d)", len(articles), len(embeddings))
	}

	if len(articles) == 0 {
		return fmt.Errorf("no articles to store")
	}

	startTime := time.Now()

	service.logger.LogService("chromadb", "store_articles", 0, map[string]interface{}{
		"articles_count": len(articles),
		"collection":     NewsCollectionName,
	}, nil)

	documents := make([]string, len(articles))
	metadatas := make([]map[string]interface{}, len(articles))

	ids := make([]string, len(articles))

	for i, article := range articles {
		documents[i] = fmt.Sprintf("%s . %s ", article.Title, article.Description)

		// Generate ID if empty
		articleID := article.ID
		if articleID == "" {
			articleID = fmt.Sprintf("article_%d_%d", time.Now().Unix(), i)
		}

		metadatas[i] = map[string]interface{}{
			"id":              articleID,
			"title":           article.Title,
			"url":             article.URL,
			"source":          article.Source,
			"author":          article.Author,
			"published_at":    article.PublishedAt.Format(time.RFC3339),
			"category":        article.Category,
			"relevance_score": article.RelevanceScore,
			"stored_at":       time.Now().Format(time.RFC3339),
		}

		ids[i] = articleID

	}

	addRequest := AddRequest{
		Documents:  documents,
		Metadatas:  metadatas,
		IDs:        ids,
		Embeddings: embeddings,
	}

	if err := service.addToCollection(ctx, NewsCollectionName, addRequest); err != nil {
		service.logger.LogService("chromadb", "store_articles", time.Since(startTime),
			map[string]interface{}{
				"articles_count": len(articles),
			}, err)

		return fmt.Errorf("Failed to store articles: %w", err)
	}

	service.logger.LogService("chromadb", "store_articles", time.Since(startTime), map[string]interface{}{
		"articles_count": len(articles),
		"collection":     NewsCollectionName,
	}, nil)

	return nil

}

func (service *ChromaDBService) addToCollection(ctx context.Context, collectionName string, addRequest AddRequest) error {
	// Get collection ID first
	collectionID, err := service.getCollectionID(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("Failed to get collection ID: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/tenants/%s/databases/%s/collections/%s/add", service.baseURL, service.tenant, service.database, collectionID)
	jsonData, err := json.Marshal(addRequest)
	if err != nil {
		return fmt.Errorf("Failed to marshall add request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("Failed to create new add to collection request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := service.client.Do(req)
	if err != nil {
		return fmt.Errorf("Failed add to collection request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Failed add to collection request: status %d, body: %s", resp.StatusCode, string(body))
	}
	return nil
}

func (service *ChromaDBService) SearchSimilarArticles(ctx context.Context, queryEmbedding []float64, topK int, filters map[string]interface{}) ([]SearchResult, error) {
	if len(queryEmbedding) == 0 {
		return nil, fmt.Errorf("query_embedding cannot be empty")
	}

	if topK <= 0 {
		topK = DefaultTopK
	}

	startTime := time.Now()

	service.logger.LogService("chromadb", "search_similar_articles", 0, map[string]interface{}{
		"top_k":       topK,
		"has_filters": len(filters) > 0,
		"collection":  NewsCollectionName,
	}, nil)

	queryRequest := QueryRequest{
		QueryEmbeddings: [][]float64{queryEmbedding},
		NResults:        topK,
		Include:         []string{"documents", "metadatas", "distances"},
	}

	if len(filters) > 0 {
		queryRequest.Where = filters
	}

	queryResponse, err := service.queryCollection(ctx, NewsCollectionName, queryRequest)
	if err != nil {
		service.logger.LogService("chromadb", "search_similar_articles", time.Since(startTime), map[string]interface{}{
			"topK": topK,
		}, err)
		return nil, fmt.Errorf("search query failed: %w", err)
	}

	results := service.convertToSearchResults(queryResponse)

	service.logger.LogService("chromadb", "search_similar_articles", time.Since(startTime), map[string]interface{}{
		"top_k":         topK,
		"results_count": len(results),
		"collection":    NewsCollectionName,
	}, nil)

	return results, nil

}

func (service *ChromaDBService) convertToSearchResults(queryResponse *QueryResponse) []SearchResult {
	var results []SearchResult

	if len(queryResponse.IDs) == 0 || len(queryResponse.IDs[0]) == 0 {
		return results
	}

	ids := queryResponse.IDs[0]
	documents := queryResponse.Documents[0]
	distances := queryResponse.Distances[0]
	metadatas := queryResponse.Metadatas[0]

	for i := 0; i < len(ids); i++ {
		metadata := metadatas[i]

		publishedAt := time.Now()
		if publishedAtStr, ok := metadata["published_at"].(string); ok {
			if parsed, err := time.Parse(time.RFC3339, publishedAtStr); err == nil {
				publishedAt = parsed
			}
		}

		relevanceScore := 0.0
		if score, ok := metadata["relevance_score"].(float64); ok {
			relevanceScore = score
		}

		article := models.NewsArticle{
			ID:             getString(metadata, "id"),
			Title:          getString(metadata, "title"),
			URL:            getString(metadata, "url"),
			Source:         getString(metadata, "source"),
			Author:         getString(metadata, "author"),
			PublishedAt:    publishedAt,
			Description:    documents[i],
			Category:       getString(metadata, "category"),
			RelevanceScore: relevanceScore,
		}

		similarity := 1.0 - distances[i]
		if similarity < 0 {
			similarity = 0
		}

		results = append(results, SearchResult{
			Document:   article,
			Similarity: similarity,
			Distance:   distances[i],
		})

	}

	return results
}

func getString(metadata map[string]interface{}, key string) string {
	if val, ok := metadata[key].(string); ok {
		return val
	}
	return ""
}

func (service *ChromaDBService) getCollectionID(ctx context.Context, collectionName string) (string, error) {
	url := fmt.Sprintf("%s/api/v2/tenants/%s/databases/%s/collections", service.baseURL, service.tenant, service.database)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("Failed to create get collections request: %w", err)
	}

	resp, err := service.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("Failed get collections request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Failed get collections request: invalid status code: %d", resp.StatusCode)
	}

	var collections []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&collections); err != nil {
		return "", fmt.Errorf("Failed to decode collections response: %w", err)
	}

	for _, col := range collections {
		if name, ok := col["name"].(string); ok && name == collectionName {
			if id, ok := col["id"].(string); ok {
				return id, nil
			}
		}
	}

	return "", fmt.Errorf("Collection %s not found", collectionName)
}

func (service *ChromaDBService) DeleteArticles(ctx context.Context, articlesIDs []string) error {
	if len(articlesIDs) == 0 {
		return fmt.Errorf("no articles to delete")
	}

	startTime := time.Now()

	service.logger.LogService("chromadb", "delete_articles", 0, map[string]interface{}{
		"articles_count": len(articlesIDs),
		"collection":     NewsCollectionName,
	}, nil)

	// Get collection ID first
	collectionID, err := service.getCollectionID(ctx, NewsCollectionName)
	if err != nil {
		return fmt.Errorf("Failed to get collection ID: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/tenants/%s/databases/%s/collections/%s/delete", service.baseURL, service.tenant, service.database, collectionID)

	deleteRequest := map[string]interface{}{
		"ids": articlesIDs,
	}

	jsonData, err := json.Marshal(deleteRequest)
	if err != nil {
		return fmt.Errorf("failed to marshal delete request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("Failed to create delete articles request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := service.client.Do(req)
	if err != nil {
		service.logger.LogService("chromadb", "delete_articles", time.Since(startTime), map[string]interface{}{
			"articles_count": len(articlesIDs),
		}, err)
		return fmt.Errorf("Failed to delete articles: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("delete failed with status: %d", resp.StatusCode)
	}

	service.logger.LogService("chromadb", "delete_articles", time.Since(startTime), map[string]interface{}{
		"articles_count": len(articlesIDs),
		"collection":     NewsCollectionName,
	}, nil)

	return nil

}

func (service *ChromaDBService) SearchByCategory(ctx context.Context, queryEmbedding []float64, category string, topK int) ([]SearchResult, error) {
	filters := map[string]interface{}{
		"category": category,
	}
	return service.SearchSimilarArticles(ctx, queryEmbedding, topK, filters)
}

func (service *ChromaDBService) SearchRecentArticles(ctx context.Context, queryEmbedding []float64, hoursBack int, topK int) ([]SearchResult, error) {
	cutoffTime := time.Now().Add(-time.Duration(hoursBack) * time.Hour)
	filters := map[string]interface{}{
		"published_at": map[string]interface{}{
			"$gte": cutoffTime.Format(time.RFC3339),
		},
	}
	return service.SearchSimilarArticles(ctx, queryEmbedding, topK, filters)
}

func (cdb *ChromaDBService) SearchBySource(ctx context.Context, queryEmbedding []float64, source string, topK int) ([]SearchResult, error) {
	filters := map[string]interface{}{
		"source": source,
	}
	return cdb.SearchSimilarArticles(ctx, queryEmbedding, topK, filters)
}

func (service *ChromaDBService) queryCollection(ctx context.Context, collectionName string, queryRequest QueryRequest) (*QueryResponse, error) {
	// Get collection ID first
	collectionID, err := service.getCollectionID(ctx, collectionName)
	if err != nil {
		return nil, fmt.Errorf("Failed to get collection ID: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/tenants/%s/databases/%s/collections/%s/query", service.baseURL, service.tenant, service.database, collectionID)
	
	// Convert v1 query request to v2 format
	v2Request := map[string]interface{}{
		"query_embeddings": queryRequest.QueryEmbeddings,
		"n_results":        queryRequest.NResults,
		"include":          queryRequest.Include,
	}
	if queryRequest.Where != nil {
		v2Request["where"] = queryRequest.Where
	}

	jsonData, err := json.Marshal(v2Request)
	if err != nil {
		return nil, fmt.Errorf("Failed to marshall query: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("Failed to create new query request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := service.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Failed query collection request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Failed query collection request: invalid status code: %d", resp.StatusCode)
	}

	var queryResponse QueryResponse
	if err := json.NewDecoder(resp.Body).Decode(&queryResponse); err != nil {
		return nil, fmt.Errorf("Failed query collection request: %w", err)
	}

	return &queryResponse, nil

}
