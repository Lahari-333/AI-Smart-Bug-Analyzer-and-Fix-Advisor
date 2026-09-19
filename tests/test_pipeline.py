"""
Unit and Integration Test Suite for AI Smart Bug Analyzer & Fix Advisor.
Validates multi-agent orchestration, triage heuristics, RAG vector retrieval,
API endpoints, and error handling boundaries.
"""

import unittest
from fastapi.testclient import TestClient
import main


class TestBugAnalyzerPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_log_analysis_agent(self):
        """Verify Agent 1 parses levels, call sites, and anomaly tokens."""
        trace = (
            "ERROR 2026-09-19 in com.service.routing.RequestDispatcher.dispatch(): "
            "NullPointerException triggered on user profile lookup"
        )
        res = main.run_log_analysis_agent(trace)
        self.assertEqual(res["detected_log_level"], "ERROR")
        self.assertIn("dispatch()", res["execution_site"])
        self.assertTrue(res["stack_trace_detected"])
        self.assertIsInstance(res["key_anomaly_tokens"], list)

    def test_triage_agent_critical(self):
        """Verify Agent 2 accurately classifies CRITICAL severity errors."""
        trace = "CRITICAL: QueuePool limit of size 10 overflow 10 reached, connection timed out"
        res = main.run_triage_agent(trace, "DB_POOL")
        self.assertEqual(res["severity"], "CRITICAL")
        self.assertEqual(res["affected_component"], "DB_POOL")

    def test_triage_agent_nullpointer(self):
        """Verify Agent 2 categorizes NullPointerException as HIGH severity."""
        trace = "NullPointerException: Cannot invoke method because object is null"
        res = main.run_triage_agent(trace, "API_GATEWAY")
        self.assertEqual(res["severity"], "HIGH")
        self.assertEqual(res["error_type"], "NullPointerException")

    def test_root_cause_and_fix_database_timeout(self):
        """Verify Agents 3 & 4 provide pool diagnosis and SQLAlchemy remediation patch."""
        trace = "sqlalchemy.exc.TimeoutError: QueuePool limit reached in execute_query()"
        rc = main.run_root_cause_agent(trace, "DB_POOL", [])
        self.assertIn("Connection Pool", rc["root_cause_summary"])
        self.assertGreaterEqual(len(rc["contributing_factors"]), 2)

        fix = main.run_fix_advisor_agent(trace, rc)
        self.assertIn("create_engine", fix["suggested_patch"])
        self.assertGreaterEqual(len(fix["remediation_steps"]), 1)
        self.assertGreaterEqual(len(fix["preventative_measures"]), 1)

    def test_root_cause_and_fix_nullpointer(self):
        """Verify Agents 3 & 4 provide null dereference diagnosis and defensive patch."""
        trace = "NullPointerException: Cannot invoke getSettings() because userProfile is null"
        rc = main.run_root_cause_agent(trace, "API_GATEWAY", [])
        self.assertIn("Null", rc["root_cause_summary"])

        fix = main.run_fix_advisor_agent(trace, rc)
        self.assertIn("Optional", fix["suggested_patch"])
        self.assertTrue(any("null" in s.lower() for s in fix["remediation_steps"]))

    def test_root_cause_and_fix_auth_expired(self):
        """Verify Agents 3 & 4 provide token expiration diagnosis and refresh patch."""
        trace = "jwt.exceptions.ExpiredSignatureError: Signature has expired in verify_token()"
        rc = main.run_root_cause_agent(trace, "AUTH_SERVICE", [])
        self.assertIn("Authentication Token", rc["root_cause_summary"])

        fix = main.run_fix_advisor_agent(trace, rc)
        self.assertIn("jwt.ExpiredSignatureError", fix["suggested_patch"])

    def test_api_analyze_bug_success(self):
        """Verify /api/v1/analyze-bug endpoint returns complete structured report."""
        payload = {
            "trace_text": "sqlalchemy.exc.TimeoutError: QueuePool limit of size 10 overflow reached",
            "component": "DB_POOL",
            "title": "Database Pool Saturation Under Peak Load"
        }
        response = self.client.post("/api/v1/analyze-bug", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("bug_id", data)
        self.assertEqual(data["bug_title"], "Database Pool Saturation Under Peak Load")
        self.assertEqual(data["priority"], "P1")
        self.assertGreaterEqual(data["confidence_score"], 0.5)
        self.assertIn("confidence_source", data)
        self.assertIn("log_analysis", data)
        self.assertIn("triage", data)
        self.assertIn("root_cause", data)
        self.assertIn("fix_suggestion", data)
        self.assertIsInstance(data["retrieved_knowledge"], list)

    def test_api_analyze_bug_empty_trace_validation(self):
        """Verify /api/v1/analyze-bug rejects empty input with HTTP 400."""
        payload = {
            "trace_text": "   ",
            "component": "DB_POOL"
        }
        response = self.client.post("/api/v1/analyze-bug", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot be empty", response.json()["detail"])

    def test_api_seed_knowledge_base(self):
        """Verify /api/v1/seed-kb populates ChromaDB benchmark records."""
        response = self.client.post("/api/v1/seed-kb")
        self.assertIn(response.status_code, [200, 503])
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertGreaterEqual(data["total_indexed"], 5)

    def test_api_run_test_suite_endpoint(self):
        """Verify /api/v1/run-tests executes all internal multi-agent tests."""
        response = self.client.post("/api/v1/run-tests")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(all(t["status"] in ["PASSED", "WARNING"] for t in data["test_results"]))

    def test_api_analytics_endpoint(self):
        """Verify /api/v1/analytics returns vector database and severity telemetry."""
        response = self.client.get("/api/v1/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_indexed_defects", data)
        self.assertIn("vector_db_status", data)
        self.assertIn("severity_distribution", data)

    def test_api_statistical_analysis_endpoint(self):
        """Verify /api/v1/statistical-analysis returns aggregate risk index."""
        response = self.client.get("/api/v1/statistical-analysis")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("system_risk_index_percentage", data)
        self.assertIn("severity_breakdown", data)

    def test_root_dashboard_html(self):
        """Verify GET / serves the interactive recruiter dashboard HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Smart Bug Analyzer & Fix Advisor", response.text)
        self.assertIn("loadSampleBug", response.text)
        self.assertIn("aibafa_analysis_history", response.text)

    def test_api_health_endpoint(self):
        """Verify /api/health returns HTTP 200, healthy status, and vector database status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "AI Smart Bug Analyzer & Fix Advisor")
        self.assertIn("version", data)
        self.assertIn("vector_database", data)
        self.assertIn("status", data["vector_database"])


if __name__ == "__main__":
    unittest.main()
