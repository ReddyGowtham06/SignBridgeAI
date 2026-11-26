"""
SignBridge AI - Phase 10B Frontend-Facing End-to-End API Workflow Tests
Exercises all 10 user workflows required for frontend-backend integration.
"""
from fastapi.testclient import TestClient


def test_phase10b_full_user_workflow(client: TestClient):
    """
    Verifies the complete 10-step user journey:
    1. Register -> 2. Login -> 3. Protected profile -> 4. Save translation ->
    5. View history -> 6. Delete history -> 7. Learning Studio lessons & progress ->
    8. Complete lesson & save score -> 9. Logout -> 10. Verify protected routes require auth.
    """
    # Step 10 (Pre-check): Protected routes require authentication
    assert client.get("/api/users/me").status_code == 401
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/learning/progress").status_code == 401

    # Step 1: Register a new account
    reg_res = client.post(
        "/api/auth/register",
        json={
            "name": "Phase10B Tester",
            "email": "phase10b@signbridge.ai",
            "password": "StrongUserPass2026!"
        }
    )
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert reg_data["user"]["email"] == "phase10b@signbridge.ai"

    # Step 2: Log in
    login_res = client.post(
        "/api/auth/login",
        json={
            "email": "phase10b@signbridge.ai",
            "password": "StrongUserPass2026!"
        }
    )
    assert login_res.status_code == 200
    tokens = login_res.json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    # Step 3: Access protected dashboard / profile
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["name"] == "Phase10B Tester"

    # Step 4: Save a translation from Sign-to-Text
    save_res = client.post(
        "/api/history",
        json={
            "source_language": "ISL",
            "translation_direction": "sign_to_text",
            "input_text": None,
            "recognized_output": "HELLO GOOD",
            "confidence": 0.91,
            "recognition_path": "static_path"
        },
        headers=headers
    )
    assert save_res.status_code == 201
    saved_id = save_res.json()["id"]

    # Step 5: View translation history
    hist_res = client.get("/api/history", headers=headers)
    assert hist_res.status_code == 200
    assert hist_res.json()["total"] == 1
    assert hist_res.json()["items"][0]["recognized_output"] == "HELLO GOOD"

    # Step 6: Delete history entry
    del_res = client.delete(f"/api/history/{saved_id}", headers=headers)
    assert del_res.status_code == 204
    assert client.get("/api/history", headers=headers).json()["total"] == 0

    # Step 7: Open Learning Studio (lessons & progress)
    lessons_res = client.get("/api/learning/lessons")
    assert lessons_res.status_code == 200
    lessons = lessons_res.json()
    assert len(lessons) >= 4
    target_lesson_id = lessons[0]["id"]

    prog_before = client.get("/api/learning/progress", headers=headers)
    assert prog_before.status_code == 200
    assert prog_before.json()["completed_lessons"] == 0

    # Step 8: Complete a lesson and save progress
    complete_res = client.post(
        "/api/learning/progress",
        json={
            "lesson_id": target_lesson_id,
            "completion_status": "completed",
            "score": 95.0
        },
        headers=headers
    )
    assert complete_res.status_code == 200
    assert complete_res.json()["completion_status"] == "completed"
    assert complete_res.json()["score"] == 95.0

    prog_after = client.get("/api/learning/progress", headers=headers)
    assert prog_after.status_code == 200
    assert prog_after.json()["completed_lessons"] == 1
    assert prog_after.json()["average_score"] == 95.0

    # Step 9: Log out
    logout_res = client.post("/api/auth/logout", headers=headers)
    assert logout_res.status_code == 200
