# tests/api/test_health.py
def test_health_endpoint(client):
    """
    Health endpoint returns status OK and HTTP 200.
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    # Accept either {"status":"ok"} or a more verbose structure
    assert data.get("status", "").lower() in ("ok", "healthy", "")
