"""
GMCR Executive PDF Report Generator
Uses fpdf2 to create branded, professional multi-page reports.
"""
from fpdf import FPDF
from datetime import datetime

# --- CORPORATE COLOR PALETTE ---
THEME = {
    "primary": (0, 102, 153),      # Deep Corporate Blue
    "secondary": (0, 150, 170),    # Teal Accent
    "text_dark": (40, 40, 40),     # Almost Black
    "text_muted": (100, 100, 100), # Grey
    "success_bg": (230, 245, 230), # Light Green
    "success_fg": (30, 130, 50),   # Dark Green
    "danger_bg": (250, 230, 230),  # Light Red
    "danger_fg": (200, 40, 40),    # Dark Red
    "row_alt": (245, 245, 250)     # Light Grey for table stripes
}

class GMCRReport(FPDF):
    """Custom PDF class with sleek GMCR branding."""

    def header(self):
        # Subtle, professional top header
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*THEME["text_muted"])
        self.cell(0, 8, 'GMCR | Global Multi-Cloud Recovery Orchestrator', 0, 1, 'C')
        
        # Sleek header line
        current_y = self.get_y()
        self.set_draw_color(*THEME["secondary"])
        self.set_line_width(0.6)
        self.line(10, current_y, 200, current_y)
        self.ln(5)

    def footer(self):
        # Clean footer with page numbers
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(*THEME["text_muted"])
        
        # Top border line for footer
        self.set_draw_color(220, 220, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Confidential & Proprietary', 0, 0, 'C')

    def section_title(self, title):
        self.ln(4)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(*THEME["primary"])
        self.cell(0, 8, title, 0, 1, 'L')
        
        # Subtle underline for sections
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def body_text(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(*THEME["text_dark"])
        self.multi_cell(0, 6, text)
        self.ln(2)

    def alert_box(self, title, message, is_breach):
        """Draws a highly visible alert box for SLA status."""
        bg_color = THEME["danger_bg"] if is_breach else THEME["success_bg"]
        fg_color = THEME["danger_fg"] if is_breach else THEME["success_fg"]
        
        self.set_fill_color(*bg_color)
        self.set_text_color(*fg_color)
        self.set_draw_color(*fg_color)
        self.set_line_width(0.4)
        
        # Box Header
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, f"  {title}", border='LTR', ln=1, fill=True)
        
        # Box Body
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, f"  {message}\n", border='LBR', fill=True)
        self.ln(4)

    def kv_table_row(self, key, value, fill=False):
        """Draws a stylized row simulating a modern table."""
        if fill:
            self.set_fill_color(*THEME["row_alt"])
        else:
            self.set_fill_color(255, 255, 255)
            
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(*THEME["text_dark"])
        self.cell(70, 8, f"  {str(key)}", border=0, fill=True)
        
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*THEME["text_muted"])
        self.cell(0, 8, str(value), border=0, ln=1, fill=True)


def generate_pdf_report(platform, prediction, physics_baseline, sla_target,
                        downtime_cost, confidence, model_choice,
                        workload_config, is_breach, cost_data=None):
    """
    Generates a beautifully formatted executive PDF report.
    Returns bytes suitable for st.download_button.
    """
    pdf = GMCRReport()
    pdf.alias_nb_pages()
    now = datetime.now().strftime("%B %d, %Y %H:%M")

    # ==========================================
    # PAGE 1: COVER
    # ==========================================
    pdf.add_page()
    pdf.ln(60)
    
    # Title Block
    pdf.set_font('Helvetica', 'B', 36)
    pdf.set_text_color(*THEME["primary"])
    pdf.cell(0, 15, 'GMCR', 0, 1, 'C')
    
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(*THEME["secondary"])
    pdf.cell(0, 10, 'Global Multi-Cloud Recovery Orchestrator', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(*THEME["text_dark"])
    pdf.cell(0, 12, 'Executive DR Readiness Report', 0, 1, 'C')
    
    pdf.ln(10)
    # Metadata Block
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(*THEME["text_muted"])
    pdf.cell(0, 8, f'Platform Targeted: {platform}', 0, 1, 'C')
    pdf.cell(0, 8, f'AI Engine Deployed: {model_choice}', 0, 1, 'C')
    pdf.cell(0, 8, f'Generated On: {now}', 0, 1, 'C')

    # ==========================================
    # PAGE 2: EXECUTIVE SUMMARY & PREDICTIONS
    # ==========================================
    pdf.add_page()
    pdf.section_title('1. Executive Summary')

    if is_breach:
        breach_mins = prediction - sla_target
        financial_loss = (breach_mins / 60) * downtime_cost
        title = "!! SLA BREACH DETECTED !!"  # <-- Removed the warning emoji
        message = (f"The predicted restore time ({prediction:.1f} min) exceeds the "
                   f"SLA target ({sla_target} min) by {breach_mins:.1f} minutes.\n"
                   f"Projected Financial Exposure: ${financial_loss:,.2f}")
    else:
        buffer_val = sla_target - prediction
        title = "SLA SECURE"  # <-- Removed the checkmark emoji
        message = (f"The predicted restore time ({prediction:.1f} min) is within the "
                   f"SLA target ({sla_target} min) with a safety buffer of {buffer_val:.1f} minutes.")
                   
    pdf.alert_box(title, message, is_breach)
    
    pdf.section_title('2. Prediction Details')
    overhead = prediction - physics_baseline
    
    # Render table with alternating row colors
    details = [
        ('Cloud Platform', platform),
        ('AI Model', model_choice),
        ('Final RTO Prediction', f'{prediction:.1f} minutes'),
        ('Physics Baseline', f'{physics_baseline:.1f} minutes'),
        ('Cloud Friction (Overhead)', f'{overhead:+.1f} minutes'),
        ('SLA Target', f'{sla_target} minutes'),
        ('Downtime Cost Rate', f'${downtime_cost:,} / hour')
    ]
    if confidence > 0:
        details.append(('AI Confidence Score (R²)', f'{confidence:.1%}'))

    for i, (k, v) in enumerate(details):
        pdf.kv_table_row(k, v, fill=(i % 2 == 0))

    # ==========================================
    # PAGE 3: CONFIGURATION & COST
    # ==========================================
    pdf.add_page()
    pdf.section_title('3. Workload Configuration')
    
    for i, (k, v) in enumerate(workload_config.items()):
        formatted_key = str(k).replace('_', ' ').title()
        pdf.kv_table_row(formatted_key, str(v), fill=(i % 2 == 0))

    if cost_data:
        pdf.ln(8)
        pdf.section_title('4. Cost Estimation')
        for i, (k, v) in enumerate(cost_data.items()):
            pdf.kv_table_row(k, str(v), fill=(i % 2 == 0))

    # ==========================================
    # PAGE 4: RECOMMENDATIONS & DISCLAIMER
    # ==========================================
    pdf.add_page()
    pdf.section_title('5. AI Orchestrator Recommendations')

    recs = []
    if is_breach:
        recs.append('CRITICAL: Current configuration will breach SLA. Immediate infrastructure scaling or tier upgrade required.')
    if physics_baseline > 0 and abs(overhead) / physics_baseline > 0.3:
        recs.append('Cloud overhead exceeds 30% of the physics baseline. Consider upgrading storage tier or network bandwidth.')
    if 0 < confidence < 0.6:
        recs.append('AI confidence is low (<60%). Submit more real-world recovery feedback to improve predictive accuracy.')
    if not recs:
        recs.append('System is healthy and highly optimized. Continue standard monitoring and feedback protocols.')

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(*THEME["text_dark"])
    for i, rec in enumerate(recs, 1):
        # Adding a bullet point style
        pdf.cell(10, 8, f"{i}.", 0, 0, 'R')
        pdf.multi_cell(0, 8, rec)
        pdf.ln(2)

    pdf.ln(15)
    pdf.section_title('6. System Disclaimer')
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(*THEME["text_muted"])
    pdf.multi_cell(0, 5,
        'This report is auto-generated by the GMCR AI Orchestrator. '
        'Predictions are modeled on historical telemetry data and may vary from actual deployment conditions. '
        'Always validate with a physical Disaster Recovery drill before relying on these metrics for strict compliance or auditing purposes.'
    )

    return bytes(pdf.output())