
import streamlit as st
import random
import time
import math
from collections import deque, defaultdict

import matplotlib.pyplot as plt

st.set_page_config(page_title="IPC Interactive Lecture (Animated)", page_icon="📡", layout="wide")

st.title("📡 Interactive IPC & RPC Lecture — Animated")
st.caption("Now with step-by-step animations for Message Passing and Reliability Protocols (R / RR / RRA).")

# ---- Sidebar navigation ----
section = st.sidebar.radio(
    "Go to",
    [
        "A) Animated Message Passing",
        "B) Animated Reliability Protocols",
        "C) Endianness & Marshalling",
        "D) RPC Pipeline (stepper)",
        "E) Message Queue Simulator",
        "F) API Style Chooser",
        "G) Quick Quiz",
    ],
)

# ---------- helpers ----------
def draw_link(ax):
    ax.plot([0.15, 0.85], [0.25, 0.25])
    ax.plot([0.15, 0.85], [0.75, 0.75])
    for x in [0.15, 0.85]:
        ax.plot([x, x], [0.25, 0.75])
    ax.text(0.1, 0.75, "Client", ha="right", va="center")
    ax.text(0.9, 0.25, "Server", ha="left", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

def animate_packets(packets, duration_s, fps=20):
    frames = int(duration_s * fps)
    for f in range(frames + 1):
        t = f / frames
        yield t

def render_scene(msgs, replies, drops, dups, reorder_events):
    st.write("**Legend**: request → •  reply → ○   (dropped messages will fade before arrival)")
    fig, ax = plt.subplots()
    draw_link(ax)
    for m in msgs:
        ax.annotate(f"{m['seq']}", (0.14, 0.76), xytext=(0.14, 0.76), fontsize=8)
    for r in replies:
        ax.annotate(f"{r['seq']}", (0.86, 0.24), xytext=(0.86, 0.24), fontsize=8)
    ph = st.empty()
    return fig, ax, ph

def move_dot(ax, x, y, filled=True, alpha=1.0):
    if filled:
        ax.plot([x], [y], marker="o", alpha=alpha)
    else:
        ax.plot([x], [y], marker="o", mfc="none", alpha=alpha)

# ---------- A) Animated Message Passing ----------
if section == "A) Animated Message Passing":
    st.header("A) Animated Message Passing")
    st.write("Visualize requests and replies moving across the network. Toggle loss, duplication, and reordering.")

    cols = st.columns(5)
    with cols[0]:
        n_msgs = st.number_input("Messages", 1, 50, 6)
    with cols[1]:
        p_drop = st.slider("Drop probability", 0.0, 1.0, 0.2, 0.01)
    with cols[2]:
        p_dup = st.slider("Duplicate probability", 0.0, 1.0, 0.1, 0.01)
    with cols[3]:
        p_reorder = st.slider("Reorder probability", 0.0, 1.0, 0.2, 0.01)
    with cols[4]:
        speed = st.slider("Animation speed", 0.2, 2.0, 1.0, 0.1)

    start = st.button("▶ Start animation")

    if start:
        random.seed(42)
        # Build outgoing messages with possible duplicates and drops
        outgoing = []
        for seq in range(1, n_msgs + 1):
            copies = 1 + (1 if random.random() < p_dup else 0)
            for c in range(copies):
                drop = random.random() < p_drop
                jitter = random.random() * p_reorder
                outgoing.append({
                    "seq": seq,
                    "t0": time.time() + jitter,
                    "drop": drop,
                    "duration": 2.0 / speed,
                    "copy": c,
                })
        # Sort by t0 to simulate wire scheduling
        outgoing.sort(key=lambda m: m["t0"])

        # Prepare replies based on messages that arrive
        replies_plan = []

        fig, ax, placeholder = render_scene(outgoing, replies_plan, p_drop, p_dup, p_reorder)

        # Animate requests one-by-one; if not dropped, schedule a reply
        for m in outgoing:
            fig, ax = plt.subplots()
            draw_link(ax)
            # Animate request
            dur = m["duration"]
            dropped = m["drop"]
            for t in animate_packets([m], dur):
                ax = plt.gca()
                draw_link(ax)
                x = 0.15 + 0.70 * t
                y = 0.75
                move_dot(ax, x, y, filled=True, alpha=1.0 if not dropped else (0.6 - 0.6 * t))
                ax.text(0.5, 0.95, f"Request seq={m['seq']} ({'duplicate' if m['copy'] else 'original'})", ha="center")
                ax.text(0.5, 0.05, f"{'DROPPED' if dropped else 'IN FLIGHT'}", ha="center")
                placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.03)
            # If arrived, send a reply back (random small delay)
            if not dropped:
                reply = {"seq": m["seq"], "duration": 2.0 / speed}
                # 50% chance to reorder reply by inserting small delay
                extra = 0.5 if random.random() < p_reorder else 0.0
                # Animate reply
                fig, ax = plt.subplots()
                draw_link(ax)
                for t in animate_packets([reply], reply["duration"] + extra):
                    ax = plt.gca()
                    draw_link(ax)
                    x = 0.85 - 0.70 * min(1.0, t)
                    y = 0.25
                    move_dot(ax, x, y, filled=False, alpha=1.0)
                    ax.text(0.5, 0.95, f"Reply for seq={reply['seq']}", ha="center")
                    placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(0.03)

        st.success("Done. Observations: UDP-like behavior allows drops, dups, and reordering; applications add sequencing and retries.")

# ---------- B) Animated Reliability Protocols ----------
elif section == "B) Animated Reliability Protocols":
    st.header("B) Animated Reliability Protocols (R, RR, RRA)")

    proto = st.radio("Protocol", ["R (request only)", "RR (request/reply + retries)", "RRA (request/reply/ack)"])
    cols = st.columns(5)
    with cols[0]:
        calls = st.number_input("Calls", 1, 50, 6)
    with cols[1]:
        p_req_loss = st.slider("Req loss", 0.0, 1.0, 0.2, 0.01)
    with cols[2]:
        p_rep_loss = st.slider("Reply/ACK loss", 0.0, 1.0, 0.2, 0.01)
    with cols[3]:
        retries = st.slider("Max retries", 0, 8, 2)
    with cols[4]:
        speed = st.slider("Speed", 0.2, 2.0, 1.0, 0.1)

    go = st.button("▶ Simulate")
    canvas = st.empty()

    def frame_request(seq, label, tfrac, y, dropped=False, hollow=False):
        fig, ax = plt.subplots()
        draw_link(ax)
        x = 0.15 + 0.70 * tfrac if y > 0.5 else 0.85 - 0.70 * tfrac
        move_dot(ax, x, y, filled=not hollow, alpha=1.0 if not dropped else (0.6 - 0.6 * tfrac))
        ax.text(0.5, 0.95, label, ha="center")
        ax.text(0.5, 0.05, f"seq={seq}", ha="center")
        canvas.pyplot(fig)
        plt.close(fig)

    if go:
        executed = 0
        duplicates = 0
        timeouts = 0
        random.seed(7)
        for seq in range(1, calls + 1):
            attempts = 0
            server_execs = 0
            acked = False
            got_reply = False

            if proto == "R (request only)":
                attempts = 1
                # animate request
                for t in animate_packets([], 1.8 / speed):
                    frame_request(seq, "Request (R)", t, 0.75, dropped=False)
                    time.sleep(0.03)
                lost = random.random() < p_req_loss
                if lost:
                    for t in animate_packets([], 0.6 / speed):
                        frame_request(seq, "Request DROPPED", t, 0.75, dropped=True)
                        time.sleep(0.03)
                else:
                    server_execs += 1
                executed += 1 if server_execs > 0 else 0
                if lost:
                    timeouts += 1

            elif proto == "RR (request/reply + retries)":
                while attempts <= retries and not got_reply:
                    attempts += 1
                    # animate request attempt
                    for t in animate_packets([], 1.2 / speed):
                        frame_request(seq, f"Request attempt {attempts} (RR)", t, 0.75)
                        time.sleep(0.03)
                    if random.random() >= p_req_loss:
                        server_execs += 1
                        # animate reply
                        lost_reply = random.random() < p_rep_loss
                        for t in animate_packets([], 1.2 / speed):
                            frame_request(seq, "Reply", t, 0.25, dropped=lost_reply, hollow=True)
                            time.sleep(0.03)
                        if not lost_reply:
                            got_reply = True
                executed += 1 if server_execs > 0 else 0
                duplicates += max(0, server_execs - 1)
                if not got_reply:
                    timeouts += 1

            else:  # RRA
                while attempts <= retries and not acked:
                    attempts += 1
                    # request
                    for t in animate_packets([], 1.0 / speed):
                        frame_request(seq, f"Request attempt {attempts} (RRA)", t, 0.75)
                        time.sleep(0.03)
                    if random.random() >= p_req_loss:
                        server_execs += 1
                        # reply
                        lost_reply = random.random() < p_rep_loss
                        for t in animate_packets([], 1.0 / speed):
                            frame_request(seq, "Reply", t, 0.25, dropped=lost_reply, hollow=True)
                            time.sleep(0.03)
                        if not lost_reply:
                            # ack
                            lost_ack = random.random() < p_rep_loss
                            for t in animate_packets([], 1.0 / speed):
                                frame_request(seq, "ACK", t, 0.75, dropped=lost_ack, hollow=False)
                                time.sleep(0.03)
                            if not lost_ack:
                                acked = True
                executed += 1 if server_execs > 0 else 0
                duplicates += max(0, server_execs - 1)
                if not acked:
                    timeouts += 1

        st.success("Simulation complete.")
        st.metric("Calls that executed at least once", executed)
        st.metric("Duplicate executions (risk!)", duplicates)
        st.metric("Client timeouts (no reply/ack)", timeouts)
        if proto.startswith("RR"):
            st.info("RR gives **at-least-once** unless you deduplicate. RRA reduces ambiguity using explicit ACKs.")

# ---------- C) Endianness & Marshalling ----------
elif section == "C) Endianness & Marshalling":
    st.header("C) Endianness & Marshalling")
    hex_in = st.text_input("Hex value (no 0x)", "07BA")
    byte_count = st.number_input("Bytes", min_value=1, max_value=8, value=2)
    if st.button("Convert"):
        try:
            val = int(hex_in, 16)
            be = val.to_bytes(byte_count, byteorder="big", signed=False).hex(" ")
            le = val.to_bytes(byte_count, byteorder="little", signed=False).hex(" ")
            st.write(f"**Big-endian**: {be}")
            st.write(f**"Little-endian**: {le}")
            st.caption("Network byte order is big-endian.")
        except Exception as e:
            st.error(f"Invalid hex/bytes: {e}")
    rep = st.radio("Representation for {year:1978, name:'Lars'}", ["JSON", "XML", "XDR/ASN.1", "Protocol Buffers", "Java Serialization"])
    pros = {
        "JSON": "Human-readable, ubiquitous, weak typing.",
        "XML": "Verbose, schemas (XSD), good for documents.",
        "XDR/ASN.1": "Strongly specified, cross-lang binary.",
        "Protocol Buffers": "Compact, schema-evolved binary.",
        "Java Serialization": "Tight with JVM, versioning issues.",
    }
    st.info(pros[rep])

# ---------- D) RPC Pipeline (stepper) ----------
elif section == "D) RPC Pipeline (stepper)":
    st.header("D) RPC Pipeline (stepper)")
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

# ---------- E) Message Queue Simulator ----------
elif section == "E) Message Queue Simulator":
    st.header("E) Message Queue Simulator")
    a, b, c = st.columns(3)
    with a:
        prod_rate = st.slider("Producer msgs/sec", 0.0, 200.0, 50.0, 1.0)
    with b:
        cons_rate = st.slider("Consumer msgs/sec", 0.0, 200.0, 40.0, 1.0)
    with c:
        duration = st.slider("Duration (sec)", 1, 120, 30)
    if st.button("Run Queue Sim"):
        backlog = 0.0
        ys = []
        for t in range(duration + 1):
            backlog = max(0.0, backlog + prod_rate - cons_rate)
            ys.append(backlog)
        st.line_chart({"backlog": ys})
        if prod_rate > cons_rate:
            st.warning("Backlog grows → scale consumers / batch / DLQ.")
        elif prod_rate < cons_rate:
            st.success("Backlog drains; capacity is sufficient.")
        else:
            st.info("Backlog stable; system at equilibrium.")

# ---------- F) API Style Chooser ----------
elif section == "F) API Style Chooser":
    st.header("F) API Style Chooser (SOAP / REST / GraphQL / gRPC)")
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
        rec.append("gRPC")
    if rec:
        st.success("Recommended fit(s): " + ", ".join(sorted(set(rec))))
    else:
        st.info("Select some constraints to see a recommendation.")

# ---------- G) Quick Quiz ----------
elif section == "G) Quick Quiz":
    st.header("G) Quick Quiz")
    def q(question, options, answer_idx, key):
        st.write("**" + question + "**")
        choice = st.radio("", options, key=key, horizontal=True)
        if st.button("Check " + key):
            if options.index(choice) == answer_idx:
                st.success("✅ Correct")
            else:
                st.error(f"❌ Not quite. Correct: {options[answer_idx]}")
    q("Which layer provides marshalling & request-reply semantics?",
      ["Application", "Middleware", "Transport", "Network"],
      1, "q1")
    q("RR + timeouts without dedup gives:",
      ["At-most-once", "At-least-once", "Exactly-once"], 1, "q2")
    q("Network byte order is:",
      ["Little-endian", "Big-endian"], 1, "q3")
