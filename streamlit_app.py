        if show_map:
            df_map = st.session_state.devices.copy()

            # --- Compute per-device risk (for radius) and pull latest SNR/loss for tooltip
            risks, snrs, losses = [], [], []
            for _, r in df_map.iterrows():
                feats = st.session_state.last_features.get(r.device_id, {})
                # risk from model
                if feats:
                    X = pd.DataFrame([feats])
                    Xs = st.session_state.scaler.transform(X)
                    Xs_df = to_df(Xs, X.columns)
                    risks.append(float(st.session_state.model.predict_proba(Xs_df)[:, 1][0]))
                else:
                    risks.append(0.0)
                # latest raw metrics for tooltip (if available)
                buf = st.session_state.dev_buf.get(r.device_id, [])
                if buf and len(buf) > 0:
                    snrs.append(float(buf[-1].get("snr", np.nan)))
                    losses.append(float(buf[-1].get("packet_loss", np.nan)))
                else:
                    snrs.append(np.nan)
                    losses.append(np.nan)

            df_map["risk"] = risks
            df_map["snr"] = snrs
            df_map["packet_loss"] = losses

            # --- Color by device type (RGBA)
            type_colors = {
                "AGV":     [0, 128, 255, 220],   # blue
                "Truck":   [255, 165, 0, 220],   # orange
                "Sensor":  [34, 197, 94, 220],   # green
                "Gateway": [147, 51, 234, 220],  # purple
            }
            df_map["fill_color"] = df_map["type"].map(type_colors)

            # --- Label text and dynamic radius (risk-aware)
            df_map["label"] = df_map.apply(lambda r: f"{r.device_id} ({r.type})", axis=1)
            df_map["radius"] = 6 + (df_map["risk"] * 16)  # 6..22 px

            # --- Layers
            layers = []

            # Device points (colored by type, size by risk)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position='[lon, lat]',
                    get_fill_color='fill_color',
                    get_radius='radius',
                    get_line_color=[0, 0, 0, 140],
                    get_line_width=1,
                    pickable=True
                )
            )

            # Device labels
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=df_map,
                    get_position='[lon, lat]',
                    get_text='label',
                    get_color=[20, 20, 20, 255],
                    get_size=12,
                    get_alignment_baseline="'top'",
                    get_pixel_offset=[0, 10],  # a little below the dot
                    pickable=False
                )
            )

            # AP marker + label
            ap_df = pd.DataFrame([{"lat": st.session_state.ap["lat"], "lon": st.session_state.ap["lon"], "name": "AP"}])
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=ap_df,
                    get_position='[lon, lat]',
                    get_fill_color='[30, 144, 255, 240]',
                    get_radius=10
                )
            )
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=ap_df.assign(label="AP"),
                    get_position='[lon, lat]',
                    get_text='label',
                    get_color=[30, 144, 255, 255],
                    get_size=14,
                    get_alignment_baseline="'bottom'",
                    get_pixel_offset=[0, -10]
                )
            )

            # Jammer marker + radius (only for jamming scenario)
            if scenario.startswith("Jamming"):
                jam = st.session_state.jammer
                jam_df = pd.DataFrame([{"lat": jam["lat"], "lon": jam["lon"], "name": "Jammer"}])
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=jam_df,
                        get_position='[lon, lat]',
                        get_fill_color='[255, 0, 0, 240]',
                        get_radius=10
                    )
                )
                layers.append(
                    pdk.Layer(
                        "TextLayer",
                        data=jam_df.assign(label="Jammer"),
                        get_position='[lon, lat]',
                        get_text='label',
                        get_color=[255, 0, 0, 255],
                        get_size=14,
                        get_alignment_baseline="'bottom'",
                        get_pixel_offset=[0, -10]
                    )
                )
                # Jammer radius ring
                angles = np.linspace(0, 2*np.pi, 60)
                circle = [{
                    "path": [[
                        jam["lon"] + meters_to_latlon_offset(CFG.jam_radius_m * math.sin(a),
                                                             CFG.jam_radius_m * math.cos(a),
                                                             st.session_state.devices.lat.mean())[1],
                        jam["lat"] + meters_to_latlon_offset(CFG.jam_radius_m * math.sin(a),
                                                             CFG.jam_radius_m * math.cos(a),
                                                             st.session_state.devices.lat.mean())[0]
                    ] for a in angles],
                    "name": "jam_radius"
                }]
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        circle,
                        get_path="path",
                        get_color=[255, 0, 0],
                        width_scale=4,
                        width_min_pixels=1,
                        opacity=0.25
                    )
                )

            # View and tooltip
            view_state = pdk.ViewState(
                latitude=float(st.session_state.devices.lat.mean()),
                longitude=float(st.session_state.devices.lon.mean()),
                zoom=14, pitch=0
            )
            tooltip = {
                "html": "<b>{device_id}</b> • {type}<br/>Risk: {risk}<br/>SNR: {snr} dB<br/>Loss: {packet_loss}%",
                "style": {"backgroundColor": "rgba(255,255,255,0.95)", "color": "#111"}
            }

            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip),
                            use_container_width=True)

            # Legend
            st.markdown(
                """
                <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#0080FF;border-radius:2px;display:inline-block;"></span> AGV
                  </span>
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#FFA500;border-radius:2px;display:inline-block;"></span> Truck
                  </span>
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#22C55E;border-radius:2px;display:inline-block;"></span> Sensor
                  </span>
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#9333EA;border-radius:2px;display:inline-block;"></span> Gateway
                  </span>
                  <span style="margin-left:auto;opacity:.8;">Dot size = risk</span>
                </div>
                """,
                unsafe_allow_html=True
            )
