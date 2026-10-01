import os
import sys
import io
import uuid
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_session_isolation():
    print("\n" + "="*60)
    print("TEST: Multi-User Session Isolation")
    print("="*60)

    session_a = f"user_alpha_{uuid.uuid4().hex[:8]}"
    session_b = f"user_beta_{uuid.uuid4().hex[:8]}"

    # 1. Check User A documents (should be 0)
    res_a_empty = client.get(f"/documents?session_id={session_a}")
    assert res_a_empty.status_code == 200
    data_a = res_a_empty.json()
    print(f"User A initially has {data_a['total_documents']} documents. (Expected: 0)")
    assert data_a["total_documents"] == 0

    # 2. User A asks greeting
    res_greet = client.post("/query", json={"query": "Hello there!", "session_id": session_a})
    assert res_greet.status_code == 200
    greet_json = res_greet.json()
    print(f"User A greeting response: {greet_json['answer'][:60]}... (Intent: {greet_json['intent']})")
    assert greet_json["intent"] == "greeting"

    # 3. User A asks document question with empty session
    res_query_empty = client.post("/query", json={"query": "Explain the UART registers in my document", "session_id": session_a})
    assert res_query_empty.status_code == 200
    empty_ans = res_query_empty.json()
    print(f"User A empty session query response: {empty_ans['answer']}")
    assert "upload" in empty_ans["answer"].lower()

    # 4. User A uploads a custom text document
    sample_text = (
        "STM32F4 UART Configuration Guide.\n"
        "The USART_BRR register controls the baud rate generation.\n"
        "To set 115200 baud at 16MHz, write 0x8B to USART_BRR.\n"
        "Interrupt priority is configured via the NVIC IP register."
    )
    file_bytes = sample_text.encode("utf-8")
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("stm32_uart.txt", io.BytesIO(file_bytes), "text/plain")},
        data={"session_id": session_a},
    )
    assert upload_res.status_code == 201
    upload_json = upload_res.json()
    print(f"User A uploaded: {upload_json['filename']} (session: {upload_json['session_id']})")

    # Process file for User A
    proc_res = client.post(f"/documents/process?session_id={session_a}&filename=stm32_uart.txt")
    assert proc_res.status_code == 200
    print(f"User A processed: {proc_res.json()['processed_documents']} ({proc_res.json()['total_chunks_added']} chunks)")

    # 5. Verify User A now has 1 document
    res_a_after = client.get(f"/documents?session_id={session_a}")
    assert res_a_after.status_code == 200
    assert res_a_after.json()["total_documents"] == 1
    print(f"User A now has: {res_a_after.json()['total_documents']} document: {[d['document_name'] for d in res_a_after.json()['documents']]}")

    # 6. CRUCIAL: Verify User B still has 0 documents!
    res_b = client.get(f"/documents?session_id={session_b}")
    assert res_b.status_code == 200
    assert res_b.json()["total_documents"] == 0
    print(f"[OK] ISOLATION VERIFIED: User B still has {res_b.json()['total_documents']} documents!")

    # 7. User A queries their document
    res_a_query = client.post("/query", json={"query": "What value should be written to USART_BRR for 115200 baud?", "session_id": session_a})
    assert res_a_query.status_code == 200
    ans_a = res_a_query.json()
    print(f"User A query answer: {ans_a['answer']}")
    print(f"User A retrieved sources: {[s['document_name'] for s in ans_a['retrieved_sources']]}")
    assert len(ans_a["retrieved_sources"]) > 0
    assert ans_a["retrieved_sources"][0]["document_name"] == "stm32_uart.txt"

    # 8. User B queries for the same thing - User B MUST NOT see User A's document!
    res_b_query = client.post("/query", json={"query": "What value should be written to USART_BRR for 115200 baud?", "session_id": session_b})
    assert res_b_query.status_code == 200
    ans_b = res_b_query.json()
    print(f"User B query answer (isolated): {ans_b['answer']}")
    assert len(ans_b["retrieved_sources"]) == 0
    print("[OK] DATA LEAK DEFENSE VERIFIED: User B cannot retrieve User A's data!")

    print("\nALL MULTI-TENANT SESSION ISOLATION TESTS PASSED!\n")

if __name__ == "__main__":
    test_session_isolation()
