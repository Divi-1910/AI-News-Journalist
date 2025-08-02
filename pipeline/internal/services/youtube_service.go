package services

import (
	"Infiya-ai-pipeline/internal/models"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"Infiya-ai-pipeline/internal/config"
	"Infiya-ai-pipeline/internal/pkg/logger"
)

type YouTubeService struct {
	apiKey  string
	client  *http.Client
	logger  *logger.Logger
	baseURL string
}

type YouTubeSearchResponse struct {
	Kind          string `json:"kind"`
	Etag          string `json:"etag"`
	NextPageToken string `json:"nextPageToken"`
	PrevPageToken string `json:"prevPageToken"`
	PageInfo      struct {
		TotalResults   int `json:"totalResults"`
		ResultsPerPage int `json:"resultsPerPage"`
	} `json:"pageInfo"`
	Items []YouTubeSearchItem `json:"items"`
}

type YouTubeSearchItem struct {
	Kind string `json:"kind"`
	Etag string `json:"etag"`
	ID   struct {
		Kind    string `json:"kind"`
		VideoID string `json:"videoId"`
	} `json:"id"`
	Snippet YouTubeSnippet `json:"snippet"`
}

type YouTubeSnippet struct {
	PublishedAt          string               `json:"publishedAt"`
	ChannelID            string               `json:"channelId"`
	Title                string               `json:"title"`
	Description          string               `json:"description"`
	Thumbnails           map[string]Thumbnail `json:"thumbnails"`
	ChannelTitle         string               `json:"channelTitle"`
	Tags                 []string             `json:"tags"`
	CategoryID           string               `json:"categoryId"`
	LiveBroadcastContent string               `json:"liveBroadcastContent"`
}

type Thumbnail struct {
	URL    string `json:"url"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
}

type YouTubeVideoResponse struct {
	Kind  string             `json:"kind"`
	Items []YouTubeVideoItem `json:"items"`
}

type YouTubeVideoItem struct {
	ID             string                `json:"id"`
	Snippet        YouTubeSnippet        `json:"snippet"`
	Statistics     YouTubeStatistics     `json:"statistics"`
	ContentDetails YouTubeContentDetails `json:"contentDetails"`
}

type YouTubeStatistics struct {
	ViewCount    string `json:"viewCount"`
	LikeCount    string `json:"likeCount"`
	CommentCount string `json:"commentCount"`
}

type YouTubeContentDetails struct {
	Duration   string `json:"duration"`
	Definition string `json:"definition"`
	Caption    string `json:"caption"`
}

func NewYouTubeService(config config.YoutubeConfig, logger *logger.Logger) (*YouTubeService, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("YouTube API key is required")
	}

	service := &YouTubeService{
		apiKey:  config.APIKey,
		client:  &http.Client{Timeout: 30 * time.Second},
		logger:  logger,
		baseURL: "https://www.googleapis.com/youtube/v3",
	}

	logger.Info("YouTube Service initialized successfully",
		"base_url", service.baseURL,
		"timeout", "30 seconds")

	return service, nil
}

// SearchNewsVideos searches for news-related videos using keywords
func (ys *YouTubeService) SearchNewsVideos(ctx context.Context, keywords []string, maxResults int) ([]models.YouTubeVideo, error) {
	if len(keywords) == 0 {
		return []models.YouTubeVideo{}, nil
	}

	// Construct news-focused query
	query := strings.Join(keywords, " ") + " news"

	ys.logger.Debug("Searching news videos",
		"keywords", keywords,
		"query", query,
		"max_results", maxResults)

	return ys.searchVideos(ctx, query, maxResults, true)
}

// SearchVideosByQuery searches for videos using a direct query string
func (ys *YouTubeService) SearchVideosByQuery(ctx context.Context, query string, maxResults int) ([]models.YouTubeVideo, error) {
	if strings.TrimSpace(query) == "" {
		return []models.YouTubeVideo{}, nil
	}

	ys.logger.Debug("Searching videos by query",
		"query", query,
		"max_results", maxResults)

	return ys.searchVideos(ctx, query, maxResults, false)
}

// searchVideos is the core search implementation
func (ys *YouTubeService) searchVideos(ctx context.Context, query string, maxResults int, newsOnly bool) ([]models.YouTubeVideo, error) {
	startTime := time.Now()

	// Build search parameters
	params := url.Values{}
	params.Set("part", "snippet")
	params.Set("q", query)
	params.Set("type", "video")
	params.Set("maxResults", strconv.Itoa(maxResults))
	params.Set("order", "relevance")
	params.Set("key", ys.apiKey)

	// Add news-specific filters
	if newsOnly {
		// Filter for recent videos (last 7 days) for news relevance
		weekAgo := time.Now().AddDate(0, 0, -7)
		params.Set("publishedAfter", weekAgo.Format(time.RFC3339))
		params.Set("videoDuration", "medium") // 4-20 minutes, good for news
		params.Set("regionCode", "US")        // Adjust based on your target audience
	}

	searchURL := fmt.Sprintf("%s/search?%s", ys.baseURL, params.Encode())

	// Execute search request
	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create YouTube search request: %w", err)
	}

	resp, err := ys.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("YouTube search request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("YouTube API returned status %d", resp.StatusCode)
	}

	// Parse response
	var searchResponse YouTubeSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResponse); err != nil {
		return nil, fmt.Errorf("failed to decode YouTube search response: %w", err)
	}

	// Convert to video models
	videos := make([]models.YouTubeVideo, 0, len(searchResponse.Items))
	for _, item := range searchResponse.Items {
		video, err := ys.convertToVideoModel(item)
		if err != nil {
			ys.logger.Warn("Failed to convert YouTube item to video model",
				"video_id", item.ID.VideoID,
				"error", err)
			continue
		}
		videos = append(videos, video)
	}

	duration := time.Since(startTime)
	ys.logger.Info("YouTube search completed",
		"query", query,
		"videos_found", len(videos),
		"duration", duration,
		"news_only", newsOnly)

	return videos, nil
}

// GetVideoDetails fetches detailed information for specific video IDs
func (ys *YouTubeService) GetVideoDetails(ctx context.Context, videoIDs []string) ([]models.YouTubeVideo, error) {
	if len(videoIDs) == 0 {
		return []models.YouTubeVideo{}, nil
	}

	startTime := time.Now()

	params := url.Values{}
	params.Set("part", "snippet,statistics,contentDetails")
	params.Set("id", strings.Join(videoIDs, ","))
	params.Set("key", ys.apiKey)

	detailsURL := fmt.Sprintf("%s/videos?%s", ys.baseURL, params.Encode())

	req, err := http.NewRequestWithContext(ctx, "GET", detailsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create YouTube details request: %w", err)
	}

	resp, err := ys.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("YouTube details request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("YouTube API returned status %d", resp.StatusCode)
	}

	// Parse response
	var videoResponse YouTubeVideoResponse
	if err := json.NewDecoder(resp.Body).Decode(&videoResponse); err != nil {
		return nil, fmt.Errorf("failed to decode YouTube video response: %w", err)
	}

	// Convert to detailed video models
	videos := make([]models.YouTubeVideo, 0, len(videoResponse.Items))
	for _, item := range videoResponse.Items {
		video, err := ys.convertDetailedVideoModel(item)
		if err != nil {
			ys.logger.Warn("Failed to convert detailed YouTube item to video model",
				"video_id", item.ID,
				"error", err)
			continue
		}
		videos = append(videos, video)
	}

	duration := time.Since(startTime)
	ys.logger.Info("YouTube video details fetched",
		"video_ids", videoIDs,
		"videos_detailed", len(videos),
		"duration", duration)

	return videos, nil
}

// convertToVideoModel converts YouTube search item to video model
func (ys *YouTubeService) convertToVideoModel(item YouTubeSearchItem) (models.YouTubeVideo, error) {
	publishedAt, err := time.Parse(time.RFC3339, item.Snippet.PublishedAt)
	if err != nil {
		ys.logger.Warn("Failed to parse video published date",
			"video_id", item.ID.VideoID,
			"date_string", item.Snippet.PublishedAt)
		publishedAt = time.Now() // Fallback to current time
	}

	// Get thumbnail URL (prefer medium, fallback to default)
	thumbnailURL := ""
	if thumb, exists := item.Snippet.Thumbnails["medium"]; exists {
		thumbnailURL = thumb.URL
	} else if thumb, exists := item.Snippet.Thumbnails["default"]; exists {
		thumbnailURL = thumb.URL
	}

	video := models.YouTubeVideo{
		ID:           item.ID.VideoID,
		Title:        item.Snippet.Title,
		Description:  item.Snippet.Description,
		ChannelID:    item.Snippet.ChannelID,
		Channel:      item.Snippet.ChannelTitle,
		ThumbnailURL: thumbnailURL,
		PublishedAt:  publishedAt,
		URL:          fmt.Sprintf("https://www.youtube.com/watch?v=%s", item.ID.VideoID),
		Tags:         item.Snippet.Tags,
		SourceType:   "youtube_video",
	}

	return video, nil
}

// convertDetailedVideoModel converts YouTube video item with statistics to video model
func (ys *YouTubeService) convertDetailedVideoModel(item YouTubeVideoItem) (models.YouTubeVideo, error) {
	publishedAt, err := time.Parse(time.RFC3339, item.Snippet.PublishedAt)
	if err != nil {
		ys.logger.Warn("Failed to parse video published date",
			"video_id", item.ID,
			"date_string", item.Snippet.PublishedAt)
		publishedAt = time.Now()
	}

	// Get thumbnail URL
	thumbnailURL := ""
	if thumb, exists := item.Snippet.Thumbnails["medium"]; exists {
		thumbnailURL = thumb.URL
	} else if thumb, exists := item.Snippet.Thumbnails["default"]; exists {
		thumbnailURL = thumb.URL
	}

	video := models.YouTubeVideo{
		ID:           item.ID,
		Title:        item.Snippet.Title,
		Description:  item.Snippet.Description,
		ChannelID:    item.Snippet.ChannelID,
		Channel:      item.Snippet.ChannelTitle,
		ThumbnailURL: thumbnailURL,
		PublishedAt:  publishedAt,
		URL:          fmt.Sprintf("https://www.youtube.com/watch?v=%s", item.ID),
		Tags:         item.Snippet.Tags,
		ViewCount:    item.Statistics.ViewCount,
		LikeCount:    item.Statistics.LikeCount,
		CommentCount: item.Statistics.CommentCount,
		Duration:     item.ContentDetails.Duration,
		SourceType:   "youtube_video",
	}

	return video, nil
}

func (ys *YouTubeService) HealthCheck(ctx context.Context) error {
	testCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	videos, err := ys.SearchVideosByQuery(testCtx, "test", 1)
	if err != nil {
		return fmt.Errorf("YouTube service health check failed: %w", err)
	}

	if len(videos) == 0 {
		ys.logger.Warn("YouTube health check returned no videos")
	}

	ys.logger.Info("YouTube service health check passed")
	return nil
}

func (ys *YouTubeService) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"service":     "youtube_service",
		"api_version": "v3",
		"base_url":    ys.baseURL,
		"timeout":     "30s",
	}
}
