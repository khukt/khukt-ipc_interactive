
import streamlit as st
import random
import time
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, Circle

st.set_page_config(page_title="IPC Lecture — PPT-style Animations", page_icon="📽️", layout="wide")

st.title("📽️ IPC & RPC — PPT-style Animations")
st.caption("Animations drawn to resemble the lecture's sequence diagrams and protocol figures.")

# -------------------------- Drawing helpers (PPT style) --------------------------

def draw_lifelines(ax, labels=("Client", "Server")):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Columns at x=2.5 and x=7.5 (like PPT diagrams)
    x_client, x_server = 2.5, 7.5
    # Header boxes
    ax.add_patch(Rectangle((x_client-1.6, 9.2), 3.2, 0.9, ec="black", fc="white"))
    ax.text(x_client, 9.65, labels[0], ha="center", va="center", fontsize=12, fontweight="bold")
    ax.add_patch(Rectangle((x_server-1.6, 9.2), 3.2, 0.9, ec="black", fc="white"))
    ax.text(x_server, 9.65, labels[1], ha="center", va="center", fontsize=12, fontweight="bold")

    # Lifelines (dashed)
    ax.plot([x_client, x_client], [9.2, 0.8], linestyle="--", color="black")
    ax.plot([x_server, x_server], [9.2, 0.8], linestyle="--", color="black")
    return x_client, x_server

def arrow(ax, x0, y0, x1, y1, text, style="solid", hollow=False, color="black"):
    lw = 1.8
    ls = "-" if style == "solid" else "--"
    head_w, head_l = 0.25, 0.35
    # FancyArrow draws a thick arrow; we emulate PPT look with a line + head
    ax.add_patch(FancyArrow(x0, y0, x1-x0, y1-y0, width=0.001, length_includes_head=True,
                            head_width=head_w, head_length=head_l, color=color, linestyle=ls, linewidth=lw,
                            fill=True))
    # Label
    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2 + 0.25
    ax.text(xm, ym, text, ha="center", va="bottom", fontsize=11, bbox=dict(fc="white", ec="none", pad=2))

def cross(ax, x, y, size=0.35, color="crimson"):
    ax.plot([x-size, x+size], [y-size, y+size], color=color, linewidth=2)
    ax.plot([x-size, x+size], [y+size, y-size], color=color, linewidth=2)

def play_frames(num, sleep):
    for i in range(num):
        yield i
        time.sleep(sleep)

# -------------------------- A) Message Passing (like slides 32–34) --------------------------

def animate_message_passing(loss=0.2, dup=0.1, reorder=0.2, msgs=5, speed=1.0):
    fig, ax = plt.subplots(figsize=(8, 5))
    xc, xs = draw_lifelines(ax)

    y = 8.6
    timeline_gap = 1.3 / max(0.35, speed)

    # Build a transmission plan (requests from client to server)
    plan = []
    rnd = random.Random(42)
    for seq in range(1, msgs+1):
        copies = 1 + (1 if rnd.random() < dup else 0)
        for c in range(copies):
            drop = rnd.random() < loss
            jitter = 0.0 if rnd.random() > reorder else rnd.random() * 0.8
            plan.append({"seq": seq, "copy": c, "drop": drop, "jitter": jitter})
    # Reorder by jitter to simulate wire scheduling
    plan.sort(key=lambda m: m["jitter"])

    frames = []
    for item in plan:
        # Request arrow animation frames (left to right)
        steps = 10
        for k in range(steps+1):
            t = k/steps
            fig, ax = plt.subplots(figsize=(8, 5))
            xc, xs = draw_lifelines(ax)
            # Already completed messages (previous arrows)
            y_completed = y
            for p in frames:
                # draw frozen frame contents
                arrow(ax, p["x0"], p["y0"], p["x1"], p["y1"], p["label"], style="solid")
            # Current moving head
            x0, y0 = xc, y
            x1, y1 = xc + (xs - xc) * t, y
            # Fading if drop
            color = "0.3" if item["drop"] else "black"
            arrow(ax, x0, y0, x1, y1, f"request #{item['seq']}", style="solid", color=color)
            st.pyplot(fig)
            plt.close(fig)
            time.sleep(0.05 / max(0.2, speed))
        # If not dropped, draw reply (right to left)
        if not item["drop"]:
            # freeze the request
            frames.append({"x0": xc, "y0": y, "x1": xs, "y1": y, "label": f"request #{item['seq']}"})
            for k in range(10+1):
                t = k/10
                fig, ax = plt.subplots(figsize=(8, 5))
                xc, xs = draw_lifelines(ax)
                for p in frames:
                    arrow(ax, p["x0"], p["y0"], p["x1"], p["y1"], p["label"], style="solid")
                x0, y0 = xs, y-0.5
                x1, y1 = xs - (xs - xc) * t, y-0.5
                arrow(ax, x0, y0, x1, y1, f"reply #{item['seq']}", style="solid")
                st.pyplot(fig)
                plt.close(fig)
                time.sleep(0.05 / max(0.2, speed))
            frames.append({"x0": xs, "y0": y-0.5, "x1": xc, "y1": y-0.5, "label": f"reply #{item['seq']}"})
        else:
            # Mark drop with a red cross near the server side
            fig, ax = plt.subplots(figsize=(8, 5))
            xc, xs = draw_lifelines(ax)
            for p in frames:
                arrow(ax, p["x0"], p["y0"], p["x1"], p["y1"], p["label"], style="solid")
            cross(ax, xs-0.4, y, size=0.28)
            ax.text(5, y+0.4, "dropped", ha="center", color="crimson")
            st.pyplot(fig)
            plt.close(fig)
        y -= timeline_gap

    st.info("This mirrors the slide style: solid arrows for messages, lifelines for Client/Server, red ✕ for drops, optional duplicates & reorder simulated.")

# -------------------------- B) Reliability Protocols (slides 36–38) --------------------------

def animate_reliability(protocol="RR", p_req_loss=0.25, p_reply_loss=0.25, retries=2, calls=4, speed=1.0):
    rnd = random.Random(7)
    for seq in range(1, calls+1):
        fig, ax = plt.subplots(figsize=(8, 5))
        xc, xs = draw_lifelines(ax)
        ax.text(5, 9.0, f"Call #{seq} — Protocol: {protocol}", ha="center", fontsize=12, fontweight="bold")

        y = 8.2
        step_h = 1.5 / max(0.35, speed)
        attempts = 0
        server_execs = 0
        done = False
        got_reply = False
        acked = False

        def anim_one_arrow(x0, y0, x1, y1, label, dropped=False):
            steps = 10
            for k in range(steps+1):
                t = k/steps
                fig, ax = plt.subplots(figsize=(8, 5))
                draw_lifelines(ax, labels=("Client", "Server"))
                ax.text(5, 9.0, f"Call #{seq} — Protocol: {protocol}", ha="center", fontsize=12, fontweight="bold")
                # draw all previous fixed arrows
                for f in fixed:
                    arrow(ax, *f["coords"], f["label"])
                # moving arrow
                color = "0.3" if dropped else "black"
                ax.add_patch(FancyArrow(x0, y0, (x1-x0)*t, (y1-y0)*t, width=0.001,
                                        length_includes_head=True, head_width=0.25, head_length=0.35,
                                        color=color))
                ax.text((x0+x1)/2, (y0+y1)/2 + 0.25, label, ha="center", bbox=dict(fc="white", ec="none", pad=2))
                st.pyplot(fig); plt.close(fig)
                time.sleep(0.05 / max(0.2, speed))

        fixed = []
        if protocol == "R":
            attempts = 1
            # Request
            lost_req = rnd.random() < p_req_loss
            anim_one_arrow(xc, y, xs, y, "request", dropped=lost_req)
            if lost_req:
                # show drop cross
                fig, ax = plt.subplots(figsize=(8, 5)); draw_lifelines(ax)
                cross(ax, xs-0.4, y, size=0.28)
                ax.text(5, y+0.4, "dropped", ha="center", color="crimson")
                st.pyplot(fig); plt.close(fig)
            else:
                server_execs += 1
            # No reply by design; client cannot know → may timeout in app layer
            st.warning("R (request-only): 'maybe' semantics. Client never knows for sure.")
        elif protocol == "RR":
            while attempts <= retries and not got_reply:
                attempts += 1
                lost_req = rnd.random() < p_req_loss
                anim_one_arrow(xc, y, xs, y, f"request (try {attempts})", dropped=lost_req)
                fixed.append({"coords": (xc, y, xs, y), "label": f"request (try {attempts})"})
                if not lost_req:
                    server_execs += 1
                    lost_reply = rnd.random() < p_reply_loss
                    anim_one_arrow(xs, y-0.5, xc, y-0.5, "reply", dropped=lost_reply)
                    if not lost_reply:
                        got_reply = True
                y -= step_h
            if got_reply:
                st.success("RR: client received reply → at-least-once (dedup needed to avoid duplicates).")
            else:
                st.warning("RR: no reply after retries → client times out (but server may have executed).")
        else:  # RRA
            while attempts <= retries and not acked:
                attempts += 1
                lost_req = rnd.random() < p_req_loss
                anim_one_arrow(xc, y, xs, y, f"request (try {attempts})", dropped=lost_req)
                fixed.append({"coords": (xc, y, xs, y), "label": f"request (try {attempts})"})
                if not lost_req:
                    server_execs += 1
                    lost_reply = rnd.random() < p_reply_loss
                    anim_one_arrow(xs, y-0.5, xc, y-0.5, "reply", dropped=lost_reply)
                    if not lost_reply:
                        lost_ack = rnd.random() < p_reply_loss
                        anim_one_arrow(xc, y-1.0, xs, y-1.0, "ack", dropped=lost_ack)
                        if not lost_ack:
                            acked = True
                y -= step_h
            if acked:
                st.success("RRA: reply + ACK observed → approximates exactly-once (with IDs & dedup).")
            else:
                st.warning("RRA: still uncertain if ACK lost — server may have executed multiple times.")

        st.info(f"Server executions for this call: {server_execs}.  Duplicates possible if multiple requests arrived.")

# -------------------------- UI --------------------------

tab1, tab2 = st.tabs(["A) Message Passing", "B) Reliability Protocols"])

with tab1:
    st.subheader("A) Message Passing — PPT sequence style")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        msgs = st.number_input("Messages", 1, 20, 6, key="mp_msgs")
    with c2:
        loss = st.slider("Loss", 0.0, 1.0, 0.2, 0.01, key="mp_loss")
    with c3:
        dup = st.slider("Duplicate", 0.0, 1.0, 0.1, 0.01, key="mp_dup")
    with c4:
        reorder = st.slider("Reorder", 0.0, 1.0, 0.2, 0.01, key="mp_reorder")
    with c5:
        speed = st.slider("Speed", 0.2, 2.0, 1.0, 0.1, key="rel_speed")

    if st.button("▶ Animate Message Passing"):
        animate_message_passing(loss=loss, dup=dup, reorder=reorder, msgs=msgs, speed=speed)

with tab2:
    st.subheader("B) Reliability Protocols — R / RR / RRA (like slides 36–38)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        protocol = st.selectbox("Protocol", ["R", "RR", "RRA"])
    with c2:
        calls = st.number_input("Calls", 1, 15, 4, key="rel_calls")
    with c3:
        p_req_loss = st.slider("Req loss", 0.0, 1.0, 0.25, 0.01, key="rel_req_loss")
    with c4:
        p_reply_loss = st.slider("Reply/ACK loss", 0.0, 1.0, 0.25, 0.01, key="rel_reply_loss")
    with c5:
        speed = st.slider("Speed", 0.2, 2.0, 1.0, 0.1)

    if st.button("▶ Animate Reliability"):
        animate_reliability(protocol, p_req_loss, p_reply_loss, retries=2, calls=calls, speed=speed)

st.divider()
st.caption("Note: These visuals intentionally mirror the slide style (two lifelines, horizontal arrows, labels). You can place this app side-by-side with the PDF for live teaching.")
