import json
import unittest

from engine import absorb, build_samples, checkpoint, markdown, new_state, parse_candidates, rank


def row(i, speaker, text, reply=None):
    return {
        "id": i, "message_id": str(i), "sender_id": speaker, "sender_name": "ExampleMember",
        "sender_nickname": "", "content_text": text, "ext_data": json.dumps({"ref_msg_id": reply}),
    }


class LearningTest(unittest.TestCase):
    def setUp(self):
        self.rows = [row(1, "a", "今晚出勤吗"), row(2, "b", "去机厅出勤", "1"),
                     row(3, "c", "舞萌出勤"), row(4, "d", "周末才出勤")]
        self.samples = build_samples(self.rows)
        self.term = {"kind": "term", "term": "出勤", "scene": "舞萌 机厅", "meaning": "去机厅打机",
                     "example": "今晚出勤吗", "evidence": [1, 2, 3, 4, 1000]}

    def test_learn_retrieve_export_and_group_isolation(self):
        state = new_state()
        items = parse_candidates(json.dumps([self.term]), self.samples)
        absorb(state, items, 4, True, 4)
        selected = rank(state, "今晚出勤不", "你最近打舞萌吗")
        self.assertEqual([state["entries"][key]["meaning"] for key in selected], ["去机厅打机"])
        self.assertEqual(rank(new_state(), "今晚出勤不", ""), [])
        self.assertIn("去机厅打机", markdown(state))
        self.assertNotIn("1000", markdown(state))
        absorb(state, items, 4, True, 4)
        self.assertEqual(state["entries"][selected[0]]["signals"], 4)

    def test_conflict_and_revision_snapshot(self):
        state = new_state()
        items = parse_candidates(json.dumps([self.term]), self.samples)
        absorb(state, items, 4, True, 4)
        key = next(iter(items))
        changed = {key: dict(items[key], meaning="上班", evidence=[5])}
        absorb(state, changed, 5, True, 4)
        self.assertEqual(rank(state, "出勤吗", ""), [])
        old = state["history"][-1]["entries"][key]
        self.assertEqual(old["status"], "active")
        self.assertEqual(old["meaning"], "去机厅打机")
        checkpoint(state)
        self.assertEqual(state["cursor"], 5)

    def test_redaction_and_grounded_style(self):
        rows = self.rows + [row(5, "e", "ExampleMember 联系 123456789 https://example.test")]
        samples = build_samples(rows)
        self.assertNotIn("ExampleMember", samples[5]["text"])
        self.assertNotIn("123456789", samples[5]["text"])
        self.assertNotIn("example.test", samples[5]["text"])
        style = dict(self.term, kind="style", evidence=[1, 3])
        self.assertEqual(parse_candidates(json.dumps([style]), samples), {})
        style["evidence"] = [1, 2]
        self.assertEqual(len(parse_candidates(json.dumps([style]), samples)), 1)

    def test_styles_cool_down(self):
        state = new_state()
        style = dict(self.term, term="临时抱佛脚", kind="style", evidence=[1, 2])
        items = parse_candidates(json.dumps([style]), self.samples)
        absorb(state, items, 4, True, 4)
        key = next(iter(items))
        self.assertEqual(rank(state, "今天舞萌打哪首", ""), [])
        state["entries"][key]["status"] = "active"
        self.assertEqual(rank(state, "今天舞萌打哪首", ""), [key])
        state["recent"] = [key, key]
        self.assertEqual(rank(state, "今天舞萌打哪首", ""), [])
