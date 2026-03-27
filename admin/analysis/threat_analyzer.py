"""
FortifAI Admin – AIAssistantManager
Wraps the Gemini generative model for cybersecurity alert analysis,
general chat, and remediation guidance.
"""

import re
from typing import List, Optional

import google.generativeai as genai

from admin.utils.config import GEMINI_API_KEY, GEMINI_MODEL


class AIAssistantManager:
    """AI assistant powered by Gemini for cybersecurity analysis."""

    SYSTEM_CONTEXT = """You are a cybersecurity expert AI assistant integrated into FortifAI,
an advanced threat detection system. You help administrators understand security alerts,
anomalies, and provide actionable remediation advice. You have expertise in:
- Network security and intrusion detection
- Malware analysis and ransomware detection
- Process behaviour analysis
- Incident response and remediation
- Threat intelligence and attribution

When analysing alerts, always follow this structure:
1. **What Happened**: Plain English explanation
2. **Why It Matters**: Risk assessment and potential impact
3. **Root Cause**: Technical analysis of the detection
4. **Recommended Actions**: Specific, prioritised remediation steps
5. **Prevention**: How to prevent similar incidents

Be concise but thorough. Use technical terms when needed but always explain them."""

    def __init__(self) -> None:
        self.available = False
        self.conversation_history: List[dict] = []

        if not GEMINI_API_KEY:
            print("⚠ GEMINI_API_KEY not set – AI Assistant disabled")
            return

        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            self.available = True
            print(f"✓ AI Assistant ready ({GEMINI_MODEL})")
        except Exception as exc:
            print(f"⚠ AI not available: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze_alert(self, alert_data: dict) -> str:
        """Generate a structured analysis for a specific alert."""
        if not self.available:
            return "AI Assistant is not available. Please check your API key."
        try:
            prompt = f"""{self.SYSTEM_CONTEXT}

Analyse this security alert from FortifAI:

Alert Details:
- Severity:    {alert_data.get('severity', 'Unknown')}
- Category:    {alert_data.get('source_category', 'Unknown')}
- Title:       {alert_data.get('title', 'N/A')}
- Description: {alert_data.get('description', 'N/A')}
- Risk Score:  {alert_data.get('risk_score', 0)}/10
- Client:      {alert_data.get('hostname', 'Unknown')}
- Timestamp:   {alert_data.get('timestamp', 'Unknown')}

Provide a comprehensive analysis following the 5-section structure above.
"""
            response = self.model.generate_content(prompt)
            return self._format_markdown(response.text)
        except Exception as exc:
            return f"Error generating analysis: {exc}"

    def chat(self, user_message: str, context_data: Optional[str] = None) -> str:
        """General chat interface with optional dashboard context."""
        if not self.available:
            return "AI Assistant is not available. Please check your API key."
        try:
            full_message = self.SYSTEM_CONTEXT + "\n\n"
            if context_data:
                full_message += f"Current Context:\n{context_data}\n\n"
            full_message += f"User Question: {user_message}"

            response = self.model.generate_content(full_message)
            formatted = self._format_markdown(response.text)

            self.conversation_history.append({
                "user":      user_message,
                "assistant": response.text,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })
            return formatted
        except Exception as exc:
            return f"Error: {exc}"

    def get_remediation_steps(self, alert_type: str, severity: str) -> List[str]:
        """Return up to 7 prioritised remediation steps for an alert type."""
        if not self.available:
            return []
        try:
            prompt = f"""As a cybersecurity expert, provide a numbered list of specific remediation
steps for this scenario:
- Alert Type: {alert_type}
- Severity:   {severity}

Format as a simple numbered list (1., 2., 3., ...) with actionable steps.
Maximum 7 steps, prioritised by urgency."""
            response = self.model.generate_content(prompt)
            text = self._format_markdown(response.text)
            steps = []
            for line in text.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith(("-", "•"))):
                    step = line.lstrip("0123456789.-•) ").strip()
                    if step:
                        steps.append(step)
            return steps[:7]
        except Exception as exc:
            print(f"Error getting remediation: {exc}")
            return []

    # ── Formatter ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_markdown(text: str) -> str:
        """Convert markdown to readable plain text for Qt text widgets."""
        text = re.sub(r"^####\s+(.+)$",  r"      \1",        text, flags=re.MULTILINE)
        text = re.sub(r"^###\s+(.+)$",   r"   \1",           text, flags=re.MULTILINE)
        text = re.sub(r"^##\s+(.+)$",    r"\n━━━ \1 ━━━",    text, flags=re.MULTILINE)
        text = re.sub(r"^#\s+(.+)$",     r"\n═══ \1 ═══",    text, flags=re.MULTILINE)
        text = re.sub(r"\*\*(.+?)\*\*",  r"[\1]",            text)
        text = re.sub(r"__(.+?)__",      r"[\1]",            text)
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
        text = re.sub(r"^\*\s+",  r"  • ", text, flags=re.MULTILINE)
        text = re.sub(r"^\-\s+",  r"  • ", text, flags=re.MULTILINE)
        text = re.sub(r"^\+\s+",  r"  • ", text, flags=re.MULTILINE)
        text = re.sub(r"^(\d+)\.\s+", r"\n\1. ", text, flags=re.MULTILINE)
        text = re.sub(r"```[\w]*\n", r"\n────────\n", text)
        text = re.sub(r"```",         r"\n────────\n", text)
        text = re.sub(r"`([^`]+)`",   r"[\1]",         text)
        text = re.sub(r"^[-*_]{3,}$", r"━" * 42, text, flags=re.MULTILINE)
        text = re.sub(r"^>\s+",       r"│ ",     text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}",      r"\n\n",   text)
        text = re.sub(r" {2,}",       r" ",       text)
        return text.strip()