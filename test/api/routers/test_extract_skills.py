def _normalize_ner_response(resp_json):
    """
    Accepts multiple possible keys and normalizes to a list-of-lists of entity dicts.
    Possible keys:
     - "results": [[{...}]]
     - "entities": [[{...}]]
     - "skills": [["Python", "FastAPI"]]
    """
    if "results" in resp_json:
        return resp_json["results"]
    if "entities" in resp_json:
        return resp_json["entities"]
    if "skills" in resp_json:
        # convert simple list of skill strings into entity-like dicts
        skills = resp_json["skills"]
        return [[{"word": s, "label": "SKILL"} for s in item] if isinstance(item, list) else [{"word": item, "label": "SKILL"}] for item in skills]
    # fallback: return any list-like value
    for v in resp_json.values():
        if isinstance(v, list):
            return v
    return []


def test_extract_skills_endpoint_returns_entities(client, sample_text):
    """
    POST /extract-skills should return detected skill entities.
    Uses DummyNER from conftest.
    """
    payload = {"texts": [sample_text]}
    resp = client.post("/extract-skills", json=payload)
    # Accept 200 OK; if endpoint returns 422/400 for validation that is allowed in some implementations,
    # but in our app it should ideally be 200.
    assert resp.status_code == 200, f"Status not OK: {resp.status_code} - {resp.text}"
    data = resp.json()
    norm = _normalize_ner_response(data)
    assert isinstance(norm, list)
    # since DummyNER returns 1 list per input, check first element
    assert len(norm) >= 1
    first = norm[0]
    assert isinstance(first, list)
    # one entity should be Python SKILL
    has_python = any(e.get("word") == "Python" for e in first if isinstance(e, dict))
    assert has_python
