"""
SignBridge AI - Translation History CRUD & Data Isolation Tests
Verifies user data isolation: users cannot see or modify each other's records.
"""
from fastapi.testclient import TestClient


def test_create_and_list_history(client: TestClient, auth_headers: dict):
    """Verifies recording a translation event and retrieving it."""
    payload = {
        "source_language": "ISL",
        "translation_direction": "sign_to_text",
        "input_text": None,
        "recognized_output": "HELLO",
        "confidence": 0.94,
        "recognition_path": "dynamic_path"
    }
    create_res = client.post("/api/history", json=payload, headers=auth_headers)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["recognized_output"] == "HELLO"
    assert created["confidence"] == 0.94
    assert created["recognition_path"] == "dynamic_path"

    # List history
    list_res = client.get("/api/history", headers=auth_headers)
    assert list_res.status_code == 200
    items = list_res.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == created["id"]
    assert items[0]["recognized_output"] == "HELLO"


def test_history_user_data_isolation(
    client: TestClient,
    auth_headers: dict,
    second_auth_headers: dict
):
    """
    CRITICAL: Verifies that User A's history cannot be read or deleted by User B.
    """
    # 1. User A records a sign translation
    res_a = client.post(
        "/api/history",
        json={"recognized_output": "GOOD", "confidence": 0.88, "recognition_path": "static_path"},
        headers=auth_headers
    )
    assert res_a.status_code == 201
    record_a_id = res_a.json()["id"]

    # 2. User B lists history - User A's record must NOT be visible
    res_b_list = client.get("/api/history", headers=second_auth_headers)
    assert res_b_list.status_code == 200
    b_items = res_b_list.json()["items"]
    assert all(item["id"] != record_a_id for item in b_items)

    # 3. User B attempts to delete User A's record - must return 404
    del_res = client.delete(f"/api/history/{record_a_id}", headers=second_auth_headers)
    assert del_res.status_code == 404

    # 4. User A can still retrieve and delete their own record
    res_a_del = client.delete(f"/api/history/{record_a_id}", headers=auth_headers)
    assert res_a_del.status_code == 204
