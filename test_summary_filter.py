
import sys
import os
from datetime import datetime, timedelta

# Mock configuration
sys.modules['agent_config_loader'] = type('agent_config_loader', (), {'load_current_agent_config': lambda x: {}})
sys.modules['agent_pool'] = type('agent_pool', (), {'submit_agent_job': lambda *a, **k: None})
sys.modules['bot'] = type('bot', (), {'QQnumber': '123', 'bot': type('Bot', (), {'api': type('API', (), {'post_private_msg': lambda **k: None})})})
sys.modules['ncatbot.core'] = type('ncatbot.core', (), {'GroupMessage': type('GM',(),{}), 'PrivateMessage': type('PM',(),{})})
sys.modules['workflows.agent_observe'] = type('workflows.agent_observe', (), {'bind_agent_event': lambda **k: lambda **j: None, 'generate_run_id': lambda: '1'})

# Add QQbot to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workflows.summary import filter_records_for_summary

def test_filter_records_with_target_date():
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    records = [
        {"ts": f"{today}T10:00:00+08:00", "message": "Today msg", "chat_type": "group", "group_id": "1"},
        {"ts": f"{yesterday}T10:00:00+08:00", "message": "Yesterday msg", "chat_type": "group", "group_id": "1"},
        {"ts": f"{yesterday}T23:59:59+08:00", "message": "Yesterday night msg", "chat_type": "group", "group_id": "1"},
        {"ts": "2020-01-01T10:00:00+08:00", "message": "Old msg", "chat_type": "group", "group_id": "1"},
    ]

    # Test with target_date = yesterday
    filtered, meta = filter_records_for_summary(records, run_mode="manual", target_date=yesterday)
    
    print(f"Testing target_date={yesterday}")
    print(f"Filtered count: {len(filtered)}")
    
    from workflows.summary import _parse_iso_dt
    for r in records:
        ts = r['ts']
        dt = _parse_iso_dt(ts)
        if dt:
            print(f"TS: {ts} -> DT: {dt} -> Local Date: {dt.astimezone().strftime('%Y-%m-%d')}")
            
    for r in filtered:
        print(f"  {r['ts']} - {r['message']}")
    
    assert len(filtered) == 2
    assert filtered[0]['message'] == "Yesterday msg"
    assert filtered[1]['message'] == "Yesterday night msg"
    assert meta['cursor_after'] == ""
    
    print("Success: target_date filtering works as expected.")

    # Test with target_date = today
    filtered_today, meta_today = filter_records_for_summary(records, run_mode="manual", target_date=today)
    print(f"Testing target_date={today}")
    print(f"Filtered count: {len(filtered_today)}")
    assert len(filtered_today) == 1
    assert filtered_today[0]['message'] == "Today msg"
    assert meta_today['cursor_after'] == ""
    print("Success: target_date today works.")

if __name__ == "__main__":
    try:
        test_filter_records_with_target_date()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
