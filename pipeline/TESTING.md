# Testing Guide for Anya AI Pipeline

## Overview

This document outlines the testing strategy and available tests for the Anya AI Pipeline.

## Test Structure

```
tests/
├── config/           # Configuration loading tests
├── handlers/         # HTTP handler tests  
├── models/          # Data model tests
├── services/        # Service layer tests
└── integration/     # End-to-end integration tests
```

## Running Tests

### Quick Test (Recommended for Development)
```bash
./test_runner.sh
```

### Unit Tests Only
```bash
make test-unit
```

### Integration Tests (Requires External Services)
```bash
make test-integration
```

### Test Coverage
```bash
make test-coverage
```

### Short Tests (Skip Integration)
```bash
make test-short
```

## Test Categories

### 1. Unit Tests
- **Models**: Test data structures, validation, and business logic
- **Handlers**: Test HTTP request/response handling
- **Services**: Test individual service functionality with mocks
- **Config**: Test configuration loading and validation

### 2. Integration Tests
- **API Tests**: Full end-to-end API testing
- **Service Integration**: Tests with real external services
- **Database Tests**: Tests with real Redis/ChromaDB connections

### 3. Performance Tests
```bash
make benchmark
```

## Test Environment Setup

### Required Environment Variables
```bash
export GEMINI_API_KEY=your_gemini_key
export NEWS_API_KEY=your_news_api_key
export REDIS_STREAMS_URL=redis://localhost:6379
export REDIS_MEMORY_URL=redis://localhost:6379
export OLLAMA_BASE_URL=http://localhost:11434
export CHROMA_DB_URL=http://localhost:9000
```

### External Services for Integration Tests
1. **Redis** (for caching and streams)
2. **ChromaDB** (for vector storage)
3. **Ollama** (for embeddings)
4. **News API** (for news data)
5. **Gemini API** (for AI processing)

## Test Examples

### Testing a New Service
```go
func TestMyNewService(t *testing.T) {
    // Setup
    service := NewMyService(testConfig, testLogger)
    
    // Test
    result, err := service.DoSomething(ctx, input)
    
    // Assert
    assert.NoError(t, err)
    assert.Equal(t, expected, result)
}
```

### Testing HTTP Handlers
```go
func TestMyHandler(t *testing.T) {
    router := setupTestRouter()
    
    req, _ := http.NewRequest("POST", "/api/test", body)
    w := httptest.NewRecorder()
    router.ServeHTTP(w, req)
    
    assert.Equal(t, http.StatusOK, w.Code)
}
```

## Mocking Strategy

### Service Mocks
- Mock external API calls
- Mock database operations
- Mock time-dependent operations

### Test Data
- Use consistent test data across tests
- Create helper functions for common test scenarios
- Use table-driven tests for multiple scenarios

## CI/CD Integration

### GitHub Actions
The pipeline includes automated testing via GitHub Actions:
- Runs on every push and PR
- Tests against multiple Go versions
- Includes coverage reporting
- Runs both unit and integration tests

### Local CI Simulation
```bash
make ci-test    # Quick CI tests
make ci-full    # Full CI pipeline with coverage
```

## Test Best Practices

### 1. Test Naming
- Use descriptive test names: `TestServiceName_MethodName_ExpectedBehavior`
- Group related tests in the same file

### 2. Test Structure
- Follow Arrange-Act-Assert pattern
- Keep tests focused and independent
- Use table-driven tests for multiple scenarios

### 3. Mocking
- Mock external dependencies
- Use interfaces for better testability
- Keep mocks simple and focused

### 4. Coverage
- Aim for >80% code coverage
- Focus on critical business logic
- Don't chase 100% coverage at the expense of test quality

## Debugging Tests

### Verbose Output
```bash
go test -v ./tests/...
```

### Run Specific Test
```bash
go test -v -run TestSpecificFunction ./tests/services/
```

### Debug with Delve
```bash
dlv test ./tests/services/ -- -test.run TestSpecificFunction
```

## Performance Testing

### Benchmarks
```bash
go test -bench=. -benchmem ./tests/...
```

### Load Testing
Use tools like `hey` or `wrk` for load testing:
```bash
hey -n 1000 -c 10 http://localhost:8080/api/v1/health
```

## Test Data Management

### Test Fixtures
- Store test data in `testdata/` directories
- Use JSON files for complex test scenarios
- Keep test data minimal and focused

### Database State
- Clean up after each test
- Use transactions that can be rolled back
- Consider using test containers for isolation

## Troubleshooting

### Common Issues
1. **Missing Environment Variables**: Ensure all required env vars are set
2. **External Service Dependencies**: Check if external services are running
3. **Port Conflicts**: Ensure test ports don't conflict with running services
4. **Race Conditions**: Use proper synchronization in concurrent tests

### Test Isolation
- Each test should be independent
- Clean up resources after tests
- Use unique identifiers for test data

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Add integration tests for new endpoints
4. Update this documentation if needed

## Monitoring Test Health

### Coverage Reports
- Generated in `coverage.html`
- Track coverage trends over time
- Focus on critical paths

### Test Performance
- Monitor test execution time
- Optimize slow tests
- Consider parallel test execution

---

For questions or issues with testing, please refer to the main project documentation or create an issue in the repository.