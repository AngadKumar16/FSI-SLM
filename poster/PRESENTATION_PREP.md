# AGU Bright STaRS — Presentation Prep

Everything you need to talk about the poster with confidence. Practice the pitches out loud until they feel natural, not memorized.

---

## 1. The 30-second hook (say this when someone walks up)

> "Hi, I'm Angad. When water or air flows past a structure — like a hydrofoil, a bridge cable, or a plane wing — it makes it vibrate, and sometimes that vibration causes it to fail. Reading that data normally takes an expert. I built a system that reads it automatically: a neural network predicts the flow, and a small language model explains what's happening in plain English — and the whole thing runs on a laptop."

Then point to the metric cards: "1,926 simulations, five kinds of vibration, no supercomputer."

*(Numbers to have ready, all measured: 1,926 cases · 107 airfoil shapes · 17 held out · Cl R² = 0.998 · 2.3% velocity-field error · milliseconds vs. ~4 minutes for the equivalent CFD solve · 11.6 M parameters · runs on one laptop.)*

---

## 2. The 2-minute walkthrough (follow the columns)

**Column 1 — the problem.**
> "FSI stands for fluid–structure interaction. The dangerous cases are things like *lock-in*, where the vortices peeling off the structure start shedding at the structure's own natural frequency, and the vibration blows up. There are five regimes we care about [point to `fig_regimes`], from calm attached flow to stall flutter. Telling them apart usually needs a trained engineer."

**Column 2 — how it works.**
> "Here's the pipeline [`fig_system`]. On the left, data comes from two places: a real PIV experiment, and a fast neural-network flow sensor. That sensor is a U-Net [`fig_unet`] — a convolutional encoder–decoder. You give it the airfoil shape, encoded as a signed-distance field [`fig_sdf_channels`], plus the angle of attack, and it predicts the whole velocity and pressure field and the lift and drag — instead of running a slow CFD simulation. All of that becomes a compact JSON summary, which a small language model reads. We fine-tuned it and gave it retrieval over 16 real FSI papers [`fig_rag`] so its answers stay anchored to the measured record."

*(Say "16 papers" — that is what is in the index. And say "anchored to the measured record", not "so it doesn't make things up": the prompt pins the feature record as the only source of truth and every output is physics-checked, but we never measured a hallucination rate, so don't claim one.)*

**Column 3 — results.**
> "We validated the flow physics on a NACA 0009 hydrofoil — you can see the correct low-pressure suction at the leading edge [`fig_flowfield`], and the lift curve it learns [`fig_cl_alpha`]. The sensor is trained across all 1,926 cases; on seventeen airfoils it has never seen, including that NACA 0009, it gets lift to R² = 0.998, drag to R² = 0.995, and the velocity field to about 2.3% error — in milliseconds instead of minutes. We also ran real RANS CFD on the held-out foil, and the velocity field transfers almost intact: 2.6% error against a solver the network never trained on. On the interpretation side, the language model gets every regime right on our test set and catches lock-in with perfect recall [`fig_confusion`] — it never misses one, though it does over-call about a quarter of them. Next we hook up the live PIV rig."

Close: "The point is a full digital twin — sensor plus interpreter — that a student can run on one laptop."

---

## 3. Talking points per figure (one line each)

- **fig_metric_cards** — "These are the headline numbers."
- **fig_twin** — "Measure the real foil once; then ask the digital copy anything."
- **fig_regimes** — "Five regimes, each with a different vibration signature."
- **fig_viv_lockin** — "Lock-in is the scary one: shedding locks onto the structure's frequency."
- **fig_system** — "The whole pipeline, and it runs on a laptop."
- **fig_unet** — "Encoder squeezes the image down, decoder builds the flow back up, skip connections keep the fine detail."
- **fig_sdf_channels** — "The network sees distance-to-the-surface, not just a black-and-white shape — that gives it a smooth signal everywhere."
- **fig_flowfield** — "Red is low pressure = suction = lift. That peak at the nose is exactly what theory predicts."
- **fig_cl_alpha** — "Lift rises with angle, then the wing stalls and it drops."
- **fig_confusion** — "Diagonal = correct. We're strong except galloping vs. stall flutter."
- **fig_dataset_scatter** — "Each regime lives in its own part of parameter space — the data is physically consistent."

---

## 4. Anticipated questions (and honest answers)

**"Is the U-Net trained yet?"**
> "Yes — 11.6 million parameters, trained on all 1,926 cases at full 128×128 resolution. On seventeen airfoils it never saw, lift comes out at R² = 0.998 and the streamwise velocity field at 2.3% error. What's still running is the *unsteady* CFD past stall, which is the step that lets me train on high-fidelity data rather than just validate against it."

**"Why a *small* language model instead of ChatGPT/GPT-4?"**
> "Three reasons: it runs offline on a laptop, it's cheap and private, and fine-tuning a small model on our specific physics makes it reliable on this narrow task. RAG covers what it doesn't know."

**"How do you stop it from hallucinating?"**
> "Three mechanisms, and I should be careful about what I've actually shown. The prompt states the FeatureRecord is the only source of truth for measured numbers, so retrieved passages can't override it. Retrieval pulls real passages from 16 FSI papers for context. And every generated interpretation is parsed and physics-checked afterwards — for instance, does the Strouhal number it reports match the frequency and velocity it reports. What I haven't done is measure a hallucination rate, so I'd say those are safeguards by construction rather than a number I can quote."

*(Judges reward this kind of precision far more than a confident overclaim. If they push: RAG in the training mix actually scored slightly worse than without, which is why the shipped adapter doesn't use it — retrieval at inference time is still available.)*

**"What is a signed-distance field and why not just a picture of the airfoil?"**
> "Every pixel stores its distance to the surface — negative inside, positive outside. Unlike a black-and-white mask, it changes smoothly everywhere, so the network gets a useful gradient even far from the wing. It's standard in this field (Guo 2016, Bhatnagar 2019)."

**"Why NACA 0009 specifically?"**
> "It's a well-studied symmetric hydrofoil with lots of reference data, so we can check our answers. We hold it out for validation and train on 106 other shapes to test generalization."

**"What's Strouhal number / reduced velocity?"** (know these cold)
> "Strouhal number St = f·c/U is the non-dimensional shedding frequency. Reduced velocity U* = U/(f_n·c) compares the flow speed to how fast the structure wants to vibrate — lock-in happens in a specific U* window."

**"Your regime accuracy is 100% — isn't that suspicious?"** (expect this one)
> "It should be, and I'd read it as a statement about my test set rather than my model. The trained adapter gets 50 out of 50 regimes right, so the task isn't discriminative at that sample size — I need a harder test set. The number that actually separates models is lock-in F1, which is 0.857: recall is perfect, so it never misses a lock-in, but precision is 0.75, so it over-calls about a quarter of them. And the rule-based baseline that generated my training labels shows the *same* bias — it over-fires lock-in too. So the model inherited that from its labels rather than inventing it, which tells me the fix is in the labelling rule, not more training."

**"What could this be used for?"**
> "Any structure in a flow: hydrofoils and marine risers, wind-turbine blades, bridge decks, aircraft wings. Anywhere you'd want to interpret vibration data quickly without a specialist on hand."

**"What was the hardest part?"** (have a genuine answer ready)
> Pick a real one: getting the CFD-to-grid sampling aligned with the SDF, or keeping the language model's outputs physically consistent, or the reproducibility/seeding.

---

## 5. Words to define before you go (so you're never caught out)

FSI · PIV · CFD · U-Net · encoder/decoder · signed-distance field · angle of attack · lift & drag coefficient (Cl, Cd) · Reynolds number · Strouhal number · reduced velocity · vortex shedding · lock-in · galloping · stall flutter · LoRA fine-tuning · RAG · MCP server.

If you can explain each in one plain sentence, you're ready.

---

## 6. Day-of checklist

- Poster printed + a few **letter-size handouts** for judges.
- **Business-card-size** slip with the repo QR and your email.
- Laptop (optional) with a **30-second live demo**: `python cli.py demo` runs the whole thing end-to-end with no model download.
- Practice the 30-second and 2-minute versions **out loud** 5+ times.
- Dress one notch up from the room. Smile, make eye contact, and let them read for a beat before you launch in.
- Have one sentence ready for each: "the coolest part," "the hardest part," "what's next."

---

## 7. Delivery tips for Bright STaRS

- Judges rotate fast — lead with the hook and the numbers, not the background.
- Match your explanation to the listener: a physics judge wants Strouhal and lock-in; a general judge wants "it reads vibration data and explains it."
- It's fine to say "that's the part I'm working on next." Honesty about status reads as maturity, not weakness.
- End every conversation by inviting a question. Curiosity is contagious.
