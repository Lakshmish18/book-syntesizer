import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import reconstruction_agent as ra  # noqa: E402


class ReconstructionAgentHelpersTests(unittest.TestCase):
    def test_parse_json_object_from_fenced_payload(self):
        payload = '```json\n{"a": 1, "b": "x"}\n```'
        parsed = ra._parse_json_object(payload)
        self.assertEqual(parsed, {"a": 1, "b": "x"})

    def test_parse_json_list_from_inline_noise(self):
        payload = 'output: ["x", "y", "z"] done'
        parsed = ra._parse_json_list(payload)
        self.assertEqual(parsed, ["x", "y", "z"])

    def test_normalize_range_caps_window_to_three_pages(self):
        self.assertEqual(ra._normalize_range("4-9"), "4-6")

    def test_canonicalize_fact_normalizes_cache_levels_relation(self):
        fact = {
            "from_entity": "cache",
            "relation": "levels",
            "to_entity": "L2 cache",
        }
        canonical = ra._canonicalize_fact(fact)
        self.assertEqual(canonical["from_entity"], "cache memory")
        self.assertEqual(canonical["relation"], "hierarchy levels")
        self.assertEqual(canonical["to_entity"], "L2 cache")

    def test_infer_cache_facts_skips_existing_relations(self):
        existing = [
            {"from_entity": "cache memory", "relation": "hierarchy levels", "to_entity": "L1 cache"},
            {"from_entity": "cache memory", "relation": "types", "to_entity": "direct mapping"},
        ]
        inferred = ra._infer_cache_level_and_type_facts(existing)
        inferred_keys = {
            (f["from_entity"].lower(), f["relation"].lower(), f["to_entity"].lower())
            for f in inferred
        }
        self.assertNotIn(("cache memory", "hierarchy levels", "l1 cache"), inferred_keys)
        self.assertNotIn(("cache memory", "types", "direct mapping"), inferred_keys)
        self.assertIn(("cache memory", "hierarchy levels", "l2 cache"), inferred_keys)
        self.assertIn(("cache memory", "types", "associative mapping"), inferred_keys)


class ReconstructionAgentIntegrationSmokeTests(unittest.TestCase):
    @patch("reconstruction_agent.knowledge_graph.save_graph", autospec=True)
    @patch("reconstruction_agent.session_memory.record_retrieval", autospec=True)
    @patch("reconstruction_agent.session_memory.init_session", autospec=True)
    @patch("reconstruction_agent.schema_validator.validate_graph", autospec=True)
    @patch("reconstruction_agent.schema_validator.infer_schema", autospec=True)
    @patch("reconstruction_agent.entity_resolver.apply_resolution", autospec=True)
    @patch("reconstruction_agent.entity_resolver.resolve_entities", autospec=True)
    @patch("reconstruction_agent.entity_resolver.extract_names_from_passages", autospec=True)
    @patch("reconstruction_agent.extract_facts_from_passages", autospec=True)
    @patch("reconstruction_agent.fetch_passages_for_reconstruction", autospec=True)
    @patch("reconstruction_agent.structural_priming", autospec=True)
    def test_run_reconstruction_smoke(
        self,
        mock_structural_priming,
        mock_fetch_passages,
        mock_extract_facts,
        mock_extract_names,
        mock_resolve_entities,
        mock_apply_resolution,
        mock_infer_schema,
        mock_validate_graph,
        mock_init_session,
        mock_record_retrieval,
        mock_save_graph,
    ):
        class FakeClient:
            documents = {"doc-1": {"name": "cache-notes.pdf"}}

        fake_client = FakeClient()
        fake_openai = object()

        mock_structural_priming.return_value = {"doc-1": ["Cache Basics"]}
        mock_fetch_passages.return_value = [
            {"doc_id": "doc-1", "pages": "1-2", "text": "Cache memory has L1 cache."}
        ]
        mock_extract_facts.return_value = [
            {
                "from_entity": "cache memory",
                "relation": "hierarchy levels",
                "to_entity": "L1 cache",
                "confidence": 0.9,
                "source": "doc-1",
            }
        ]
        mock_extract_names.return_value = ["cache memory", "L1 cache"]
        mock_resolve_entities.return_value = {}
        mock_apply_resolution.side_effect = lambda text, _entity_map: text
        mock_infer_schema.return_value = {"required_relations": ["hierarchy levels", "types"]}
        mock_validate_graph.return_value = {"complete": True, "missing_aspects": []}
        mock_init_session.return_value = {"session_id": "s1"}

        result = ra.run_reconstruction(
            query="Explain cache memory",
            doc_ids=["doc-1"],
            client=fake_client,
            openai_client=fake_openai,
        )

        self.assertIn("graph", result)
        self.assertIn("validation", result)
        self.assertTrue(result["validation"].get("complete"))
        self.assertGreaterEqual(len(result["graph"].get("edges", [])), 1)
        self.assertIn("schema", result)
        self.assertIn("entity_map", result)
        mock_record_retrieval.assert_not_called()
        self.assertGreaterEqual(mock_save_graph.call_count, 1)


if __name__ == "__main__":
    unittest.main()
