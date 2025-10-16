# ---------- Governance (EU AI Act / GDPR / NIS2) — FULL FEATURED
with tab_governance:
    _init_gov_state()
    ss = st.session_state

    st.header("Governance, Transparency & Compliance")

    subtabs = st.tabs([
        "Overview",
        "Policies & Controls",
        "Data & Privacy",
        "Model & Dataset Cards",
        "Risk Register",
        "Audit & Exports",
        "Ops Settings"
    ])

    # --- Overview
    with subtabs[0]:
        st.subheader("AI System Classification (EU AI Act)")
        st.markdown(
            "Categorized as a **limited-risk AI system**: real-time anomaly detection with advisory output; "
            "**no autonomous enforcement**. All impactful actions require human confirmation."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model version", CFG.model_version)
        di = pd.DataFrame(ss.incidents) if "incidents" in ss and ss.incidents else pd.DataFrame()
        c2.metric("Incidents (session)", 0 if di.empty else len(di))
        al_df = pd.DataFrame(ss.audit_log) if ss.audit_log else pd.DataFrame()
        c3.metric("Decisions logged", 0 if al_df.empty else len(al_df))
        c4.metric("Controls enabled", sum(bool(v) for v in ss.gov_controls.values()))

        st.markdown("#### Standards & Acts mapping")
        mapping = pd.DataFrame([
            {"Framework": "EU AI Act", "Topic": "Transparency & human oversight", "Status": "Covered (demo)", "Evidence": "Incident views + audit log"},
            {"Framework": "GDPR", "Topic": "Data minimization & privacy by design", "Status": "Covered (demo)", "Evidence": "Synthetic data; settings"},
            {"Framework": "NIS2", "Topic": "Security risk management", "Status": "Planned/Assumed", "Evidence": "Secure transport note, integrity hash"},
            {"Framework": "CRA", "Topic": "Security-by-design, vulnerability handling", "Status": "Planned", "Evidence": "Versioning, exports"},
            {"Framework": "ISO/IEC 42001", "Topic": "AI management system", "Status": "In progress", "Evidence": "Risk register, lifecycle plan"},
            {"Framework": "ISO/IEC 27001/27701", "Topic": "InfoSec & Privacy", "Status": "In progress", "Evidence": "Role matrix, retention controls"},
        ])
        st.dataframe(mapping, use_container_width=True, height=220)

    # --- Policies & Controls
    with subtabs[1]:
        st.subheader("Controls checklist")
        cols = st.columns(4)
        keys = list(ss.gov_controls.keys())
        for i, k in enumerate(keys):
            with cols[i % 4]:
                ss.gov_controls[k] = st.checkbox(k.replace("_", " ").title(), value=ss.gov_controls[k])

        st.markdown("#### Role & Permission Matrix")
        st.dataframe(ss.gov_role_matrix, use_container_width=True, height=260)
        st.caption("Map personas to permissions. Hook this to your auth layer in production.")

        st.markdown("#### Lifecycle Plan (high level)")
        st.write("- Drift monitoring & false-positive review (weekly).")
        st.write("- Threshold tuning (as needed) and periodic retraining (quarterly).")
        st.write("- Versioning with rollback plan.")

    # --- Data & Privacy
    with subtabs[2]:
        st.subheader("Data Inventory")
        st.dataframe(ss.gov_data_inventory, use_container_width=True, height=260)
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.gov_retention_days_incidents = st.number_input("Incident retention (days)", 30, 1825, ss.gov_retention_days_incidents, 1)
        with c2:
            ss.gov_pseudonymization = st.checkbox("Pseudonymization enabled", ss.gov_pseudonymization)
        with c3:
            lawful_basis = st.selectbox("GDPR lawful basis (deployment)", ["— Demo —", "Legitimate interests (Art.6.1.f)", "Public interest (Art.6.1.e)", "Contract (Art.6.1.b)"], index=0)
        st.info("Demo uses **synthetic** data. For live deployment, ensure DPIA (Art.35), data minimization, and purpose limitation are documented.")

        st.markdown("#### DPIA — Draft Template (exportable)")
        di_markdown = f"""# DPIA (Draft Template)

**Scope**: Site anomaly screening (RF, access, GNSS, integrity).  
**Controller**: Your Org • **Model version**: {CFG.model_version}  
**Lawful Basis**: {lawful_basis}  
**Data categories**: Technical telemetry (no PII by design).  
**Risks**: False positives; misclassification under novel attacks.  
**Mitigations**: Human oversight, threshold tuning, monitoring, retraining, secure transport, retention {ss.gov_retention_days_incidents} days.  
**Rights & safeguards**: Access, rectification, deletion processes; privacy by design; pseudonymization: {"ON" if ss.gov_pseudonymization else "OFF"}.
"""
        _download_bytes(di_markdown, "DPIA_draft.md", "⬇️ Download DPIA (draft)")

    # --- Model & Dataset Cards
    with subtabs[3]:
        st.subheader("Model Card")
        mc = f"""# Model Card — TRUST DEMO
**Version**: {CFG.model_version}  
**Intended use**: Site-level anomaly screening; advisory only.  
**Limits**: Not for autonomous enforcement or user surveillance.  
**Known failure modes**: RF congestion may cause false positives; unknown attack classes.  
**Evaluation**: Threshold-based; qualitative expert review.  
**Ethical considerations**: Human-in-the-loop, transparency, contestability.
"""
        st.code(mc, language="markdown")
        _download_bytes(mc, f"model_card_{CFG.model_version}.md", "⬇️ Model Card (MD)")

        st.subheader("Datasheet for Datasets")
        ds = """# Datasheet — Telemetry (synthetic)
**Composition**: Synthetic RF/access/GNSS/integrity proxies.  
**Collection**: Procedural generation; no user data.  
**Preprocessing**: Normalization & signal proxies.  
**Uses**: Demo/training of anomaly screens.  
**Distribution**: Internal demo only.
"""
        st.code(ds, language="markdown")
        _download_bytes(ds, "datasheet_telemetry_synthetic.md", "⬇️ Datasheet (MD)")

    # --- Risk Register (editable)
    with subtabs[4]:
        st.subheader("Risk Register")
        st.dataframe(ss.gov_risk_register, use_container_width=True, height=280)
        with st.expander("Add new risk"):
            r1, r2 = st.columns([3,1])
            with r1: new_risk = st.text_input("Risk description")
            with r2: new_sev = st.selectbox("Severity", ["Low","Medium","High"], index=1)
            r3, r4, r5 = st.columns([1,2,1])
            with r3: new_lik = st.selectbox("Likelihood", ["Low","Medium","High"], index=1)
            with r4: new_mit = st.text_input("Mitigation")
            with r5: new_owner = st.text_input("Owner", value="—")
            if st.button("Add risk"):
                if new_risk:
                    ss.gov_risk_register = pd.concat([
                        ss.gov_risk_register,
                        pd.DataFrame([{
                            "Risk": new_risk, "Severity": new_sev, "Likelihood": new_lik,
                            "Mitigation": new_mit or "TBD", "Owner": new_owner or "TBD", "Status": "New"
                        }])
                    ], ignore_index=True)
                    st.success("Risk added.")

        _download_bytes(ss.gov_risk_register.to_csv(index=False), "risk_register.csv", "⬇️ Export CSV", mime="text/csv")

    # --- Audit & Exports
    with subtabs[5]:
        st.subheader("Audit Trail & Evidence")
        di = pd.DataFrame(ss.incidents) if ss.incidents else pd.DataFrame()
        al_df = pd.DataFrame(ss.audit_log) if ss.audit_log else pd.DataFrame()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Incidents (session)**")
            st.dataframe(di, use_container_width=True, height=250)
        with c2:
            st.markdown("**Decisions**")
            if not al_df.empty:
                al_view = al_df.copy()
                al_view["created_at"] = pd.to_datetime(al_view["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                al_view["decided_at"] = pd.to_datetime(al_view["decided_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(al_view, use_container_width=True, height=250)
            else:
                st.info("No human decisions logged yet.")

        # Bulk exports
        st.markdown("#### Bulk Exports")
        _download_bytes(di.to_csv(index=False) if not di.empty else "", "incidents.csv", "⬇️ Incidents (CSV)", mime="text/csv")
        _download_bytes(di.to_json(orient="records", indent=2) if not di.empty else "[]", "incidents.json", "⬇️ Incidents (JSON)", mime="application/json")
        _download_bytes(al_df.to_csv(index=False) if not al_df.empty else "", "audit_log.csv", "⬇️ Audit Log (CSV)", mime="text/csv")
        _download_bytes(al_df.to_json(orient="records", indent=2) if not al_df.empty else "[]", "audit_log.json", "⬇️ Audit Log (JSON)", mime="application/json")

        st.caption("Exports include timestamps and are suitable for compliance archiving and external audit ingestion.")

    # --- Ops Settings
    with subtabs[6]:
        st.subheader("Operational Settings (affect governance artifacts)")
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.gov_log_level = st.selectbox("Log level", ["Debug","Info","Warning","Error"], index=["Debug","Info","Warning","Error"].index(ss.gov_log_level))
        with c2:
            ss.gov_hash_algo = st.selectbox("Evidence hash", ["SHA-256","SHA-512"], index=0 if ss.gov_hash_algo=="SHA-256" else 1)
        with c3:
            cfg_thresh = st.slider("Default incident threshold", 0.30, 0.95, float(CFG.threshold), 0.01)
            # Note: we don't overwrite CFG.threshold live here to avoid upsetting current pipeline;
            # use this value when regenerating governance exports if you want it to drive evidence docs.
        st.write("Changes here reflect in **DPIA**, **Model Card**, and evidence exports.")
