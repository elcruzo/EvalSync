"""
Test Result Reporters for EvalSync
Generate comprehensive reports in multiple formats
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import html


class BaseReporter:
    """Base class for test result reporters"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate report from test results"""
        raise NotImplementedError


class HTMLReporter(BaseReporter):
    """Generate comprehensive HTML test reports"""
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate HTML report"""
        html_content = self._generate_html_content(results)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_content(self, results: Dict[str, Any]) -> str:
        """Generate complete HTML report content"""
        summary = results.get('summary', {})
        suite_summaries = results.get('suite_summaries', {})
        detailed_results = results.get('detailed_results', {})
        
        # Generate CSS styles
        css_styles = """
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                background: #f5f5f5;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 40px 20px; 
                border-radius: 10px; 
                margin-bottom: 30px; 
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header .subtitle { opacity: 0.9; font-size: 1.2em; }
            .summary-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .metric-card { 
                background: white; 
                padding: 25px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                text-align: center;
            }
            .metric-value { font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }
            .metric-label { color: #666; font-size: 0.9em; }
            .passed { color: #28a745; }
            .failed { color: #dc3545; }
            .warning { color: #ffc107; }
            .suite-section { 
                background: white; 
                margin-bottom: 30px; 
                border-radius: 10px; 
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .suite-header { 
                background: #f8f9fa; 
                padding: 20px; 
                border-bottom: 1px solid #dee2e6;
                cursor: pointer;
            }
            .suite-header:hover { background: #e9ecef; }
            .suite-title { font-size: 1.4em; font-weight: bold; }
            .suite-stats { margin-top: 10px; }
            .stat-badge { 
                display: inline-block; 
                padding: 4px 12px; 
                border-radius: 20px; 
                font-size: 0.8em; 
                font-weight: bold; 
                margin-right: 10px;
            }
            .badge-passed { background: #d4edda; color: #155724; }
            .badge-failed { background: #f8d7da; color: #721c24; }
            .badge-total { background: #d1ecf1; color: #0c5460; }
            .suite-content { padding: 20px; display: none; }
            .suite-content.active { display: block; }
            .test-table { width: 100%; border-collapse: collapse; }
            .test-table th, .test-table td { 
                padding: 12px; 
                text-align: left; 
                border-bottom: 1px solid #dee2e6; 
            }
            .test-table th { 
                background: #f8f9fa; 
                font-weight: bold; 
                position: sticky; 
                top: 0;
            }
            .test-row.passed { background: #f8fff9; }
            .test-row.failed { background: #fff5f5; }
            .status-indicator { 
                padding: 4px 8px; 
                border-radius: 4px; 
                font-size: 0.8em; 
                font-weight: bold; 
            }
            .error-details { 
                background: #fff3cd; 
                border: 1px solid #ffeaa7; 
                border-radius: 4px; 
                padding: 10px; 
                margin-top: 10px; 
                font-family: monospace; 
                font-size: 0.85em; 
                white-space: pre-wrap;
            }
            .performance-chart { 
                margin: 20px 0; 
                padding: 20px; 
                background: #f8f9fa; 
                border-radius: 5px; 
            }
            .footer { 
                text-align: center; 
                padding: 20px; 
                color: #666; 
                font-size: 0.9em;
                margin-top: 40px; 
            }
            .collapsible { max-height: 100px; overflow: hidden; }
            .collapsible.expanded { max-height: none; }
            .expand-btn { 
                color: #007bff; 
                cursor: pointer; 
                text-decoration: underline; 
                font-size: 0.85em;
            }
        </style>
        """
        
        # JavaScript for interactivity
        javascript = """
        <script>
            function toggleSuite(suiteId) {
                const content = document.getElementById(suiteId);
                content.classList.toggle('active');
            }
            
            function expandError(element) {
                element.classList.toggle('expanded');
                const btn = element.nextElementSibling;
                if (element.classList.contains('expanded')) {
                    btn.textContent = 'Show less';
                } else {
                    btn.textContent = 'Show more';
                }
            }
            
            // Auto-expand failed test suites
            document.addEventListener('DOMContentLoaded', function() {
                const failedSuites = document.querySelectorAll('.suite-section.has-failures .suite-content');
                failedSuites.forEach(suite => suite.classList.add('active'));
            });
        </script>
        """
        
        # Build HTML content
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>EvalSync Test Report</title>",
            css_styles,
            "</head>",
            "<body>",
            "<div class='container'>",
            
            # Header
            self._generate_header(results),
            
            # Summary metrics
            self._generate_summary_section(summary),
            
            # Suite details
            self._generate_suites_section(suite_summaries, detailed_results),
            
            # Footer
            "<div class='footer'>",
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by EvalSync",
            "</div>",
            
            "</div>",
            javascript,
            "</body>",
            "</html>"
        ]
        
        return "\n".join(html_parts)
    
    def _generate_header(self, results: Dict[str, Any]) -> str:
        """Generate header section"""
        summary = results.get('summary', {})
        status = "PASSED" if results.get('status') == 'passed' else "FAILED"
        status_class = "passed" if status == "PASSED" else "failed"
        
        return f"""
        <div class='header'>
            <h1>🧪 EvalSync Test Report</h1>
            <div class='subtitle'>
                Test Status: <span class='{status_class}' style='font-weight: bold;'>{status}</span>
                • Duration: {results.get('duration_seconds', 0):.1f}s
                • Tests: {summary.get('total_tests', 0)}
            </div>
        </div>
        """
    
    def _generate_summary_section(self, summary: Dict[str, Any]) -> str:
        """Generate summary metrics section"""
        total_tests = summary.get('total_tests', 0)
        passed_tests = summary.get('passed', 0)
        failed_tests = summary.get('failed', 0)
        pass_rate = summary.get('pass_rate', 0) * 100
        avg_duration = summary.get('average_duration_ms', 0)
        
        return f"""
        <div class='summary-grid'>
            <div class='metric-card'>
                <div class='metric-value'>{total_tests}</div>
                <div class='metric-label'>Total Tests</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value passed'>{passed_tests}</div>
                <div class='metric-label'>Passed</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value failed'>{failed_tests}</div>
                <div class='metric-label'>Failed</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value {"passed" if pass_rate >= 90 else "warning" if pass_rate >= 70 else "failed"}'>{pass_rate:.1f}%</div>
                <div class='metric-label'>Pass Rate</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{avg_duration:.1f}ms</div>
                <div class='metric-label'>Avg Duration</div>
            </div>
        </div>
        """
    
    def _generate_suites_section(self, suite_summaries: Dict[str, Any], detailed_results: Dict[str, Any]) -> str:
        """Generate test suites section"""
        suites_html = []
        
        for suite_name, suite_summary in suite_summaries.items():
            suite_results = detailed_results.get(suite_name, [])
            has_failures = suite_summary.get('failed', 0) > 0
            
            suite_html = f"""
            <div class='suite-section {"has-failures" if has_failures else ""}'>
                <div class='suite-header' onclick='toggleSuite("{suite_name}_content")'>
                    <div class='suite-title'>{suite_name}</div>
                    <div class='suite-stats'>
                        <span class='stat-badge badge-total'>{suite_summary.get('total', 0)} total</span>
                        <span class='stat-badge badge-passed'>{suite_summary.get('passed', 0)} passed</span>
                        <span class='stat-badge badge-failed'>{suite_summary.get('failed', 0)} failed</span>
                        <span style='color: #666; font-size: 0.9em;'>
                            • {suite_summary.get('pass_rate', 0) * 100:.1f}% pass rate
                            • {suite_summary.get('duration_ms', 0):.1f}ms duration
                        </span>
                    </div>
                </div>
                <div id='{suite_name}_content' class='suite-content'>
                    {self._generate_test_table(suite_results)}
                </div>
            </div>
            """
            suites_html.append(suite_html)
        
        return "\n".join(suites_html)
    
    def _generate_test_table(self, test_results: List[Dict[str, Any]]) -> str:
        """Generate test results table"""
        if not test_results:
            return "<p>No test results available.</p>"
        
        table_rows = []
        
        for result in test_results:
            status = result.get('status', 'unknown')
            status_class = 'passed' if status == 'passed' else 'failed'
            
            error_details = ""
            if result.get('error_message'):
                error_msg = html.escape(result['error_message'])
                stack_trace = html.escape(result.get('stack_trace', ''))
                
                error_details = f"""
                <div class='error-details collapsible' onclick='expandError(this)'>
                    <strong>Error:</strong> {error_msg}
                    {f'<br><br><strong>Stack Trace:</strong><br>{stack_trace}' if stack_trace else ''}
                </div>
                <div class='expand-btn' onclick='expandError(this.previousElementSibling)'>Show more</div>
                """
            
            row = f"""
            <tr class='test-row {status_class}'>
                <td>{html.escape(result.get('test_name', ''))}</td>
                <td>
                    <span class='status-indicator {status_class}'>{status.upper()}</span>
                </td>
                <td>{result.get('duration_ms', 0):.1f}ms</td>
                <td>
                    {html.escape(result.get('metadata', {}).get('category', 'unknown'))}
                </td>
                <td>
                    {error_details if error_details else 'N/A'}
                </td>
            </tr>
            """
            table_rows.append(row)
        
        return f"""
        <table class='test-table'>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Category</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
        """


class JSONReporter(BaseReporter):
    """Generate JSON test reports for programmatic consumption"""
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate JSON report"""
        # Add report metadata
        report_data = {
            'report_metadata': {
                'generator': 'EvalSync',
                'version': '1.0.0',
                'generated_at': datetime.now().isoformat(),
                'format': 'json'
            },
            **results
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)


class JUnitReporter(BaseReporter):
    """Generate JUnit XML reports for CI/CD integration"""
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate JUnit XML report"""
        root = ET.Element('testsuites')
        
        # Add overall attributes
        summary = results.get('summary', {})
        root.set('tests', str(summary.get('total_tests', 0)))
        root.set('failures', str(summary.get('failed', 0)))
        root.set('errors', '0')
        root.set('time', str(results.get('duration_seconds', 0)))
        root.set('name', 'EvalSync Test Suite')
        root.set('timestamp', datetime.now().isoformat())
        
        # Add test suites
        suite_summaries = results.get('suite_summaries', {})
        detailed_results = results.get('detailed_results', {})
        
        for suite_name, suite_summary in suite_summaries.items():
            suite_element = ET.SubElement(root, 'testsuite')
            suite_element.set('name', suite_name)
            suite_element.set('tests', str(suite_summary.get('total', 0)))
            suite_element.set('failures', str(suite_summary.get('failed', 0)))
            suite_element.set('errors', '0')
            suite_element.set('time', str(suite_summary.get('duration_ms', 0) / 1000))
            
            # Add individual test cases
            suite_results = detailed_results.get(suite_name, [])
            for result in suite_results:
                testcase = ET.SubElement(suite_element, 'testcase')
                testcase.set('name', result.get('test_name', ''))
                testcase.set('classname', suite_name)
                testcase.set('time', str(result.get('duration_ms', 0) / 1000))
                
                # Add failure information if test failed
                if result.get('status') != 'passed':
                    if result.get('error_type') == 'AssertionError':
                        failure = ET.SubElement(testcase, 'failure')
                        failure.set('message', result.get('error_message', ''))
                        failure.set('type', 'AssertionError')
                        failure.text = result.get('stack_trace', '')
                    else:
                        error = ET.SubElement(testcase, 'error')
                        error.set('message', result.get('error_message', ''))
                        error.set('type', result.get('error_type', 'Error'))
                        error.text = result.get('stack_trace', '')
        
        # Write XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(self.output_path, encoding='utf-8', xml_declaration=True)


class CSVReporter(BaseReporter):
    """Generate CSV reports for data analysis"""
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate CSV report"""
        import csv
        
        detailed_results = results.get('detailed_results', {})
        
        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'suite_name', 'test_name', 'status', 'duration_ms', 
                'category', 'error_type', 'error_message'
            ])
            
            # Write data rows
            for suite_name, suite_results in detailed_results.items():
                for result in suite_results:
                    writer.writerow([
                        suite_name,
                        result.get('test_name', ''),
                        result.get('status', ''),
                        result.get('duration_ms', 0),
                        result.get('metadata', {}).get('category', ''),
                        result.get('error_type', ''),
                        result.get('error_message', '').replace('\n', ' ') if result.get('error_message') else ''
                    ])


def create_reporter(format_type: str, output_path: str) -> BaseReporter:
    """Factory function to create reporters"""
    reporters = {
        'html': HTMLReporter,
        'json': JSONReporter,
        'junit': JUnitReporter,
        'csv': CSVReporter
    }
    
    reporter_class = reporters.get(format_type.lower())
    if not reporter_class:
        raise ValueError(f"Unknown report format: {format_type}")
    
    return reporter_class(output_path)