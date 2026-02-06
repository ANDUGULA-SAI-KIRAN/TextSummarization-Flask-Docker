import unittest
import sys
import os
import json
from unittest.mock import patch
from datetime import datetime
import html

# 1. FIX PATHING: This ensures the test can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. CACHE LOGIC: Tell the test exactly where the models live
# This prevents the test from creating a new 'model_cache' inside the tests folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TRANSFORMERS_CACHE'] = os.path.join(BASE_DIR, "model_cache")

from src.app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_summarize_endpoint_success(self):
        payload = {
            "documents": [
                "Artificial intelligence is transforming industries.",
                "It enables machines to perform tasks requiring human intelligence."
            ]
        }
        # We can mock the summarizer to speed up tests and avoid loading model
        with patch('src.app.summarize_text') as mock_summarize:
            mock_summarize.return_value = "AI transforms industries and enables machines."
            
            response = self.app.post('/summarize', 
                                     data=json.dumps(payload),
                                     content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('summary', data)
            self.assertEqual(data['summary'], "AI transforms industries and enables machines.")

    def test_summarize_endpoint_invalid_input(self):
        # Missing documents
        response = self.app.post('/summarize', 
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 422)

        # Invalid documents type
        response = self.app.post('/summarize', 
                                 data=json.dumps({"documents": "not a list"}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 422)
        
        # Empty documents list
        response = self.app.post('/summarize', 
                                 data=json.dumps({"documents": []}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 422)

    def test_model_loading_logic(self):
        """
        Verify that the real Summarizer can initialize. 
        This is the 'Real' test that checks your cache path.
        """
        from src.summarizer import Summarizer
        
        # We patch the actual inference so we don't run the heavy math, 
        # but we let the _load_model() run to verify the folder path is correct.
        with patch.object(Summarizer, '_summarize_chunk', return_value="mock summary"):
            summarizer = Summarizer()
            # If this doesn't crash, your CACHE_DIR logic is correct!
            self.assertIsNotNone(summarizer.device)
            print(f"\n✓ Summarizer initialized on {summarizer.device}")

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with custom result
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate HTML report
    report_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'report.html')
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Report - API Tests</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .summary-box {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .summary-box h3 {{
                margin: 0 0 10px 0;
                color: #555;
                font-size: 14px;
            }}
            .summary-box .number {{
                font-size: 32px;
                font-weight: bold;
                margin: 0;
            }}
            .passed {{
                color: #27ae60;
            }}
            .failed {{
                color: #e74c3c;
            }}
            .skipped {{
                color: #f39c12;
            }}
            .test-cases {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .test-case {{
                padding: 15px;
                border-bottom: 1px solid #ecf0f1;
                display: flex;
                align-items: center;
            }}
            .test-case:last-child {{
                border-bottom: none;
            }}
            .test-status {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                margin-right: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }}
            .test-status.passed {{
                background-color: #27ae60;
            }}
            .test-status.failed {{
                background-color: #e74c3c;
            }}
            .test-status.error {{
                background-color: #c0392b;
            }}
            .test-info {{
                flex: 1;
            }}
            .test-name {{
                font-weight: bold;
                margin: 0 0 5px 0;
            }}
            .test-message {{
                color: #7f8c8d;
                font-size: 12px;
                margin: 0;
            }}
            .footer {{
                text-align: center;
                color: #7f8c8d;
                margin-top: 20px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Test Report - API Tests</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-box">
                <h3>Total Tests</h3>
                <p class="number">{result.testsRun}</p>
            </div>
            <div class="summary-box">
                <h3>Passed</h3>
                <p class="number passed">{result.testsRun - len(result.failures) - len(result.errors)}</p>
            </div>
            <div class="summary-box">
                <h3>Failed</h3>
                <p class="number failed">{len(result.failures)}</p>
            </div>
            <div class="summary-box">
                <h3>Errors</h3>
                <p class="number failed">{len(result.errors)}</p>
            </div>
        </div>
        
        <h2>Test Cases</h2>
        <div class="test-cases">
    """
    
    # Add passed tests
    passed_count = result.testsRun - len(result.failures) - len(result.errors)
    for i in range(result.testsRun):
        # Check if test passed (not in failures or errors)
        is_failed = any(test[0] == suite._tests[i % len(suite._tests[0])] if hasattr(suite, '_tests') else False for test, _ in result.failures)
        is_error = any(test[0] == suite._tests[i % len(suite._tests[0])] if hasattr(suite, '_tests') else False for test, _ in result.errors)
    
    # Add failures
    for test, traceback in result.failures:
        html_content += f"""
        <div class="test-case">
            <div class="test-status failed">✗</div>
            <div class="test-info">
                <p class="test-name">{html.escape(str(test))}</p>
                <p class="test-message">{html.escape(traceback.split(chr(10))[0])}</p>
            </div>
        </div>
        """
    
    # Add errors
    for test, traceback in result.errors:
        html_content += f"""
        <div class="test-case">
            <div class="test-status error">!</div>
            <div class="test-info">
                <p class="test-name">{html.escape(str(test))}</p>
                <p class="test-message">{html.escape(traceback.split(chr(10))[0])}</p>
            </div>
        </div>
        """
    
    # Add passed tests
    for test in suite:
        if not any(t[0] == test for t, _ in result.failures) and not any(t[0] == test for t, _ in result.errors):
            html_content += f"""
        <div class="test-case">
            <div class="test-status passed">✓</div>
            <div class="test-info">
                <p class="test-name">{html.escape(str(test))}</p>
                <p class="test-message">Test passed successfully</p>
            </div>
        </div>
        """
    
    html_content += """
        </div>
        
        <div class="footer">
            <p>Test report generated by unittest</p>
        </div>
    </body>
    </html>
    """
    
    # Write report to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ HTML report generated: {report_path}")
