# run_concurrent_test.py
import threading, requests, time

def user_session(user_id, scenario):
    base = "http://localhost:7860"
    r = requests.post(f"{base}/reset", json={"task_name": scenario, "seed": user_id})
    eid = r.json()["episode_id"]
    print(f"User {user_id} ({scenario}) episode_id: {eid}")
    for step in range(5):
        r = requests.post(f"{base}/step", json={
            "episode_id": eid,
            "action": {"action_type": "skip"}
        })
        day = r.json()["observation"]["current_day"]
        print(f"User {user_id} step {step+1}: day={day}")
        time.sleep(0.1)

threads = [
    threading.Thread(target=user_session, args=(i, s))
    for i, s in enumerate(["easy_sprint", "medium_sprint", "hard_sprint"])
]
for t in threads: t.start()
for t in threads: t.join()
# Expected: each user's day counter increments independently
# If days jump: shared state bug