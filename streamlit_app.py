
import streamlit as st
import random
import time
import math
from collections import deque, defaultdict

st.set_page_config(page_title="IPC Interactive Lecture", page_icon="🧩", layout="wide")

st.title("🧩 Interactive IPC & RPC Lecture")
st.caption("Hands-on activities covering sockets, marshalling, RPC, reliability, queues, SOAP vs REST, and more.")

# ---- Sidebar navigation ----
section = st.sidebar.radio(
    "Go to",
    [
        "1) Foundations & Stack",
        "2) Sockets & Message Passing",
        "3) Data Representation & Endianness",
        "4) RPC Pipeline Simulator",
        "5) Reliability Protocols: R / RR / RRA",
        "6) Message Queue Simulator",
        "7) SOAP vs REST vs GraphQL Chooser",
        "8) API Gateway Sketchpad",
        "9) Quick Quiz & Exit Ticket",
    ],
)

# ---- 1) Foundations & Stack ----
if section == "1) Foundations & Stack":
    st.header("1) Foundations & Stack Overview")

    st.subheader("Layer Mapping Activity")
    st.write(
        "Pick an example protocol or artifact for each layer. The goal is to see how application semantics ride on lower layers."
    )

    with st.form("layer_form"):
        app = st.selectbox(
            "Application Layer (examples)",
            ["HTTP", "FTP", "SOAP", "gRPC", "Custom App Protocol"],
        )
        pres = st.selectbox(
            "Presentation Layer (data representation)",
            ["JSON", "XML", "XDR", "ASN.1", "Java Serialization"],
        )
        trans = st.selectbox("Transport Layer", ["TCP (stream)", "UDP (datagram)"])
        net = st.selectbox("Network Layer", ["IP", "IPv6", "Both"])
        link = st.selectbox("Link Layer", ["Ethernet", "Wi‑Fi", "Others"])
        submit = st.form_submit_button("Show Stack")

    if submit:
        st.success("Your stack:")
        st.markdown(
            "**Application** → {app}  \n"
            "**Presentation** → {pres}  \n"
            "**Transport** → {trans}  \n"
            "**Network** → {net}  \n"
            "**Link** → {link}"
        .format(app=app, pres=pres, trans=trans, net=net, link=link))
        st.info(
            "Observation: IPC features (RPC, marshalling, request‑reply) sit above transport (TCP/UDP) but below applications."
        )

    st.divider()
    st.subheader("Concept Nuggets")
    st.markdown(
        "- **IPC ≠ Transport**: IPC abstractions (RPC/RMI, events, message queues) rely on transport but add semantics.\n"
        "- **Request‑Reply** underpins many patterns; marshalling ensures both ends agree on data layout."
    )

# ---- 2) Sockets & Message Passing ----
elif section == "2) Sockets & Message Passing":
    st.header("2) Sockets & Message Passing")

    st.subheader("UDP Echo Playground (simulated)")
    st.write(
        "Simulate a UDP echo exchange with tunable **drop**, **duplication**, and **reordering** to see common pitfalls."
    )
    cols = st.columns(4)
    with cols[0]:
        n_msgs = st.number_input("Messages to send", min_value=1, max_value=200, value=20, step=1)
    with cols[1]:
        p_drop = st.slider("Drop probability", 0.0, 1.0, 0.1, 0.01)
    with cols[2]:
        p_dup = st.slider("Duplicate probability", 0.0, 1.0, 0.05, 0.01)
    with cols[3]:
        p_reorder = st.slider("Reorder probability", 0.0, 1.0, 0.1, 0.01)

    if st.button("Run Simulation"):
        sent = list(range(1, n_msgs + 1))
        in_flight = []
        log = []

        for seq in sent:
            if random.random() < p_drop:
                log.append(f"Client → Server: **{seq}** DROPPED")
                continue

            # maybe duplicate
            copies = 1 + (1 if random.random() < p_dup else 0)
            for _ in range(copies):
                # add jitter to simulate reordering
                jitter = random.randint(0, int(100 * p_reorder))
                in_flight.append((time.time() + jitter / 1000.0, seq))

        # deliver
        in_flight.sort(key=lambda x: x[0])
        delivered_order = [seq for _, seq in in_flight]

        # echo back
        responses = delivered_order[:]
        random.shuffle(responses)  # server sends independently; not guaranteed order

        st.write("### Client Log")
        for entry in log[:200]:
            st.write("• " + entry)
        if len(log) > 200:
            st.caption(f"... and {len(log)-200} more")

        st.write("### Server Received (order)")
        st.code(", ".join(map(str, delivered_order)) or "(none)")

        st.write("### Client Received Replies (order)")
        st.code(", ".join(map(str, responses)) or "(none)")

        lost = [s for s in sent if s not in delivered_order]
        dups = len(delivered_order) - len(set(delivered_order))

        st.metric("Unique delivered", len(set(delivered_order)))
        st.metric("Lost", len(lost))
        st.metric("Duplicates (received by server)", dups)

        st.info(
            "Takeaway: UDP is **unreliable & unordered**. Applications add reliability (e.g., sequence numbers, retransmissions)."
        )

    st.divider()
    st.subheader("Addressing Quick Check")
    ip = st.text_input("Destination IP", "138.37.94.248")
    port = st.number_input("Port", 0, 65535, 80)
    st.write(f"**Socket address** → `{ip}:{port}`")

# ---- 3) Data Representation & Endianness ----
elif section == "3) Data Representation & Endianness":
    st.header("3) Data Representation & Endianness")

    st.subheader("Endianness Converter")
    hex_in = st.text_input("Hex value (no 0x)", "07BA")
    byte_count = st.number_input("Bytes", min_value=1, max_value=8, value=2)
    if st.button("Convert"):
        try:
            val = int(hex_in, 16)
            be = val.to_bytes(byte_count, byteorder="big", signed=False)
            le = val.to_bytes(byte_count, byteorder="little", signed=False)
            st.write(f"**Big‑endian bytes**: {be.hex(' ')}")
            st.write(f"**Little‑endian bytes**: {le.hex(' ')}")
            st.caption("Network byte order is big‑endian.")
        except Exception as e:
            st.error(f"Invalid hex or byte size: {e}")

    st.divider()
    st.subheader("Marshalling Thought Experiment")
    st.markdown(
        "Pick a representation for the object `{year: 1978, name: 'Lars'}` and discuss trade‑offs."
    )
    rep = st.radio("Choose", ["JSON", "XML", "XDR/ASN.1", "Protocol Buffers", "Java Serialization"])
    pros = {
        "JSON": "Human‑readable, ubiquitous, weak typing.",
        "XML": "Verbose, schemas (XSD), good for documents.",
        "XDR/ASN.1": "Strongly specified, cross‑lang binary.",
        "Protocol Buffers": "Compact, schema‑evolved binary.",
        "Java Serialization": "Tight with JVM, versioning issues.",
    }
    st.info(pros[rep])

# ---- 4) RPC Pipeline Simulator ----
elif section == "4) RPC Pipeline Simulator":
    st.header("4) RPC Pipeline Simulator")

    st.write("Step through the phases of a remote call.")

    phase = st.select_slider(
        "Phase",
        options=[
            "Client: call()",
            "Client stub: marshal",
            "Runtime: send",
            "Server stub: unmarshal",
            "Server: execute",
            "Server stub: marshal reply",
            "Runtime: send reply",
            "Client stub: unmarshal reply",
            "Client: return",
        ],
        value="Client: call()",
    )

    explanations = {
        "Client: call()": "App code invokes a method as if local.",
        "Client stub: marshal": "Parameters packed into external form (type, byte order).",
        "Runtime: send": "Request flows over transport (TCP/UDP).",
        "Server stub: unmarshal": "Reconstruct parameters for server implementation.",
        "Server: execute": "Business logic runs; side effects may occur.",
        "Server stub: marshal reply": "Result packaged for transport.",
        "Runtime: send reply": "Reply transmitted back to client.",
        "Client stub: unmarshal reply": "Result reconstructed in client representation.",
        "Client: return": "Call completes; caller receives value/exception.",
    }
    st.success(explanations[phase])

    st.subheader("Failure Injection")
    c1, c2, c3 = st.columns(3)
    with c1:
        drop_req = st.checkbox("Drop request")
    with c2:
        drop_reply = st.checkbox("Drop reply")
    with c3:
        type_mismatch = st.checkbox("Type mismatch")

    st.write("**What happens?**")
    if drop_req:
        st.warning("Client will timeout waiting for reply (unless retries).")
    if drop_reply:
        st.warning("Server executed, but client may retry → duplicates unless idempotent.")
    if type_mismatch:
        st.error("Unmarshalling fails: version skew or schema mismatch.")

# ---- 5) Reliability Protocols ----
elif section == "5) Reliability Protocols: R / RR / RRA":
    st.header("5) Reliability Protocols Simulator")

    st.write(
        "Compare request patterns under loss. Toggle probabilities and see delivery semantics."
    )
    col = st.columns(4)
    with col[0]:
        trials = st.number_input("Calls", 1, 200, 50)
    with col[1]:
        p_loss = st.slider("Loss probability", 0.0, 1.0, 0.2, 0.01)
    with col[2]:
        p_reply_loss = st.slider("Reply loss probability", 0.0, 1.0, 0.2, 0.01)
    with col[3]:
        max_retries = st.slider("Max retries (RR/RRA)", 0, 10, 3)

    proto = st.radio("Protocol", ["R (request only)", "RR (request/reply)", "RRA (request/reply/ack)"])

    if st.button("Simulate"):
        executed = 0
        duplicates = 0
        timeouts = 0
        for i in range(trials):
            if proto == "R (request only)":
                # One-shot, no retry
                if random.random() >= p_loss:
                    executed += 1  # server executes
                else:
                    timeouts += 1
            elif proto == "RR (request/reply)":
                # Retries until reply or max_retries
                sent = 0
                got_reply = False
                server_execs = 0
                while sent <= max_retries and not got_reply:
                    sent += 1
                    if random.random() >= p_loss:
                        server_execs += 1  # request arrived; server executed
                        if random.random() >= p_reply_loss:
                            got_reply = True
                executed += 1 if server_execs > 0 else 0
                duplicates += max(0, server_execs - 1)
                if not got_reply:
                    timeouts += 1
            else:  # RRA
                sent = 0
                got_reply = False
                server_execs = 0
                acked = False
                while sent <= max_retries and not acked:
                    sent += 1
                    if random.random() >= p_loss:
                        server_execs += 1
                        # reply may be lost
                        if random.random() >= p_reply_loss:
                            got_reply = True
                            # ack may also be lost? model via reply_loss again
                            if random.random() >= p_reply_loss:
                                acked = True
                executed += 1 if server_execs > 0 else 0
                duplicates += max(0, server_execs - 1)
                if not acked:
                    timeouts += 1

        st.metric("Calls that executed at least once", executed)
        st.metric("Duplicate executions (risk!)", duplicates)
        st.metric("Client timeouts (no reply/ack)", timeouts)

        if proto.startswith("RR"):
            st.info("RR provides **at‑least‑once** semantics unless you deduplicate. RRA approximates **exactly‑once** at the cost of extra messages/state.")

# ---- 6) Message Queue Simulator ----
elif section == "6) Message Queue Simulator":
    st.header("6) Message Queue Simulator")

    st.write("Model a simple 1‑producer / 1‑consumer queue and watch backlog growth or drain.")

    a, b, c = st.columns(3)
    with a:
        prod_rate = st.slider("Producer msgs/sec", 0.0, 200.0, 50.0, 1.0)
    with b:
        cons_rate = st.slider("Consumer msgs/sec", 0.0, 200.0, 40.0, 1.0)
    with c:
        duration = st.slider("Duration (sec)", 1, 120, 30)

    if st.button("Run Queue Sim"):
        backlog = 0.0
        backlog_points = []
        for t in range(duration + 1):
            backlog = max(0.0, backlog + prod_rate - cons_rate)
            backlog_points.append({"t": t, "backlog": backlog})
        st.line_chart({ "backlog": [p["backlog"] for p in backlog_points] })
        if prod_rate > cons_rate:
            st.warning("Backlog grows → need scaling / batching / DLQ.")
        elif prod_rate < cons_rate:
            st.success("Backlog drains; capacity is sufficient.")
        else:
            st.info("Backlog stable; system at equilibrium.")

# ---- 7) SOAP vs REST vs GraphQL Chooser ----
elif section == "7) SOAP vs REST vs GraphQL Chooser":
    st.header("7) SOAP vs REST vs GraphQL Chooser")

    st.write("Pick constraints; get a recommendation.")

    needs = st.multiselect(
        "Constraints / needs",
        [
            "Strong contract & WS-Security",
            "Human-readable payloads",
            "Tight typing / codegen",
            "Binary efficiency",
            "Single endpoint / flexible queries",
            "Easy caching via HTTP",
            "Streaming updates",
            "Legacy B2B integration",
            "Mobile-friendly payload shaping",
        ],
    )

    rec = []
    if "Strong contract & WS-Security" in needs or "Legacy B2B integration" in needs:
        rec.append("SOAP")
    if "Human-readable payloads" in needs or "Easy caching via HTTP" in needs:
        rec.append("REST")
    if "Single endpoint / flexible queries" in needs or "Mobile-friendly payload shaping" in needs:
        rec.append("GraphQL")
    if "Binary efficiency" in needs or "Streaming updates" in needs:
        rec.append("gRPC (RPC over HTTP/2)")

    if rec:
        st.success("Recommended fit(s): " + ", ".join(sorted(set(rec))))
    else:
        st.info("Select some constraints to see a recommendation.")

# ---- 8) API Gateway Sketchpad ----
elif section == "8) API Gateway Sketchpad":
    st.header("8) API Gateway Sketchpad")

    st.write("Plan cross‑cutting concerns for clients and services.")

    # Simple tag inputs using text and split
    clients_raw = st.text_input("Client types (comma-separated)", "web, mobile")
    services_raw = st.text_input("Backend services (comma-separated)", "catalog, orders, accounts")

    clients = [c.strip() for c in clients_raw.split(",") if c.strip()]
    services = [s.strip() for s in services_raw.split(",") if s.strip()]

    checks = st.multiselect(
        "Policies at the gateway",
        ["AuthN/AuthZ", "Rate limiting", "Request shaping", "Protocol translation", "Caching", "Observability"],
        default=["AuthN/AuthZ", "Rate limiting", "Observability"],
    )
    st.write("**Summary**")
    st.json({"clients": clients, "services": services, "gateway_policies": checks})

# ---- 9) Quick Quiz & Exit Ticket ----
elif section == "9) Quick Quiz & Exit Ticket":
    st.header("9) Quick Quiz & Exit Ticket")

    def q(question, options, answer_idx, key):
        st.write("**" + question + "**")
        choice = st.radio("", options, key=key, horizontal=True)
        if st.button("Check " + key):
            if options.index(choice) == answer_idx:
                st.success("✅ Correct")
            else:
                st.error(f"❌ Not quite. Correct: {options[answer_idx]}")

    q("Which layer provides marshalling & request‑reply semantics?",
      ["Application", "Middleware", "Transport", "Network"],
      1, "q1")

    q("UDP gives you:", ["Reliable, in-order delivery", "Unreliable, unordered datagrams", "Message queues", "Exactly‑once semantics"], 1, "q2")

    q("RR + timeouts without dedup gives:", ["At‑most‑once", "At‑least‑once", "Exactly‑once"], 1, "q3")

    st.subheader("Exit Ticket")
    takeaway = st.text_area("One concept you understand well + one you want to revisit")
    if st.button("Save Exit Ticket"):
        # In a real deployment you'd persist this; here we keep it ephemeral.
        st.success("Saved locally (session state). Thanks!")
