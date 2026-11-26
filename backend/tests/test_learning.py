"""
SignBridge AI - Learning Studio Tests
Tests curriculum lesson querying, progress updates, and scoring.
"""
from fastapi.testclient import TestClient


def test_list_lessons(client: TestClient):
    """Verifies that default curriculum lessons are seeded and returned."""
    res = client.get("/api/learning/lessons")
    assert res.status_code == 200
    lessons = res.json()
    assert len(lessons) >= 4
    lesson_ids = [l["id"] for l in lessons]
    assert "lesson-01-greetings" in lesson_ids
    assert "lesson-02-static-foundations" in lesson_ids
    assert "lesson-03-body-parts" in lesson_ids
    assert "lesson-04-emergency" in lesson_ids


def test_learning_progress_lifecycle(client: TestClient, auth_headers: dict):
    """Verifies querying progress summary and updating lesson completion."""
    # 1. Initial progress summary should show 0 completed lessons
    res_init = client.get("/api/learning/progress", headers=auth_headers)
    assert res_init.status_code == 200
    init_data = res_init.json()
    assert init_data["completed_lessons"] == 0
    assert init_data["total_lessons"] >= 4

    # 2. Record completion for lesson 1 with score 95.0
    res_record = client.post(
        "/api/learning/progress",
        json={
            "lesson_id": "lesson-01-greetings",
            "completion_status": "completed",
            "score": 95.0
        },
        headers=auth_headers
    )
    assert res_record.status_code == 200
    prog = res_record.json()
    assert prog["lesson_id"] == "lesson-01-greetings"
    assert prog["completion_status"] == "completed"
    assert prog["score"] == 95.0

    # 3. Verify progress summary now reflects the completion
    res_after = client.get("/api/learning/progress", headers=auth_headers)
    assert res_after.status_code == 200
    after_data = res_after.json()
    assert after_data["completed_lessons"] == 1
    assert after_data["average_score"] == 95.0
