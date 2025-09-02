# tests/api/test_classify.py
import json


def _normalize_pred_response(resp_json):
    """
    Accept multiple possible response shapes:
    - {"predictions": [...]}
    - {"prediction": "..."}
    - {"predicted": [...]}
    - {"category": "..."}
    Return a list of predicted labels.
    """
    if "predictions" in resp_json:
        return resp_json["predictions"]
    if "prediction" in resp_json:
        p = resp_json["prediction"]
        return p if isinstance(p, list) else [p]
    if "predicted" in resp_json:
        return resp_json["predicted"]
    if "category" in resp_json:
        return [resp_json["category"]]
    # fallback: return any string values in the payload
    return [v for v in resp_json.values() if isinstance(v, str)]


def test_classify_endpoint_returns_prediction(client, sample_text):
    """
    POST /classify should return a prediction list or equivalent.
    """
    payload = {"texts": [sample_text]}
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 200, f"Status not OK: {resp.status_code} - {resp.text}"
    data = resp.json()
    preds = _normalize_pred_response(data)
    assert isinstance(preds, list)
    assert len(preds) == 1
    # dummy label from DummyTFIDF in conftest is "DummyLabel"
    assert preds[0] == "DummyLabel"
