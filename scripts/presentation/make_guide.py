"""
make_guide.py — writes docs/PRESENTATION_GUIDE.md from build/numbers.json, the
same numbers the deck uses, so the spoken script and the slides always agree.

Usage (normally via scripts/presentation/build.sh):
    python scripts/presentation/make_guide.py scripts/presentation/build/numbers.json docs/PRESENTATION_GUIDE.md
"""
import json, sys

N = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
M, S, D, R = N["means"], N["sig"], N["deltas"], N["red"]
ex = N["examples"]
TL = {"clear_blue": "clear blue", "green_turbid": "green turbid", "deep_lowlight": "deep low-light", "hazy_bright": "hazy bright"}
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
sg = lambda x, d=2: ("+" if x >= 0 else "−") + f"{abs(x):.{d}f}"
cv = N["alpha_cv"]
import math
axis_min = math.floor((min(cv["fixed"]["psnr"], cv["pred"]["psnr"], cv["oracle"]["psnr"]) - 0.5) * 2) / 2
share = round(100 * (cv["pred"]["psnr"] - cv["fixed"]["psnr"]) / (cv["oracle"]["psnr"] - cv["fixed"]["psnr"]))
red_order = sorted(N["red_types"], key=lambda t: -N["red_types"][t]["best"])
own_red = [t for t, v in N["red_types"].items() if v["decision"] == "per-type"]
own_prof = [t for t, v in N["routing_types"].items() if v == "per-type"]
gp = N["global_full_profile"]
rounds = N.get("rounds") or {"rounds": [], "converged": False}
nround = len(rounds["rounds"])
cmp_a, cmp_b = N["compare_examples"]
weak = N["weak_example"].replace(".png", "")
clahe_p = S["clahe_only"]["psnr"]
rt = N["runtime"]
boundary = round(100 * N["selectors"]["argmax nr_combined"]["boundary"])
sel = N["selectors"]
post_fix = N.get("rounds") is not None  # calibrate_all.py (with the gamma fix) writes calibration_rounds.json
gamma_tail = (", and after the last fix we re-ran the entire calibration and evaluation." if post_fix else
              ". The gamma fix is in the code; the calibration and evaluation are being re-run with it, so the numbers on these slides predate it.")
gamma_dir = ": values below 1 brighten" if post_fix else ""
calib_line = (("stages 1–2 alternated, converged after " if rounds["converged"] else "stages 1–2 alternated, stopped after ")
              + f"{nround} round{'s' if nround != 1 else ''}" if post_fix else
              "one pass (stage 1, 2, then 3); the re-run with scripts/calibrate_all.py and the gamma fix is pending")
worse_than_fixed = all(sel[k]["psnr"] < sel["fixed alpha=0.5"]["psnr"] for k in sel if k.startswith("argmax"))

def join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

psnr_level = clahe_p["eff"] == "negligible" or clahe_p["p"] >= 0.05
share_below = round(100 * N["share_psnr_below_clahe"])
edge_names = {"wb_gain_cap": "WB cap", "gamma_strength": "gamma ×", "clahe_clip_scale": "CLAHE ×", "dcp_omega": "ω"}
edges = [f"{edge_names[k]} {v:g}" for k, v in N["grid_edges"].items()]
red_largest = R["psnr"]["mean_diff"] > max(D["psnr"]["fusion_optimizer_alone (row4-row3)"], D["psnr"]["routing_alone (row5-row3)"])
psnr_vs_clahe_line = (
    f"we are level on PSNR ({sg(clahe_p['d'])} dB, {clahe_p['eff']}{'' if clahe_p['p'] < 0.05 else ', not significant'}) "
    f"and {sg(S['clahe_only']['ssim']['d'], 3)} higher in SSIM ({S['clahe_only']['ssim']['eff']} effect)."
    if psnr_level else

    f"we are {f2(-clahe_p['d'])} dB lower in PSNR — a {clahe_p['eff']} effect — but {sg(S['clahe_only']['ssim']['d'], 3)} in SSIM. "
    "Our dehazed stream shifts overall brightness relative to the moderate UIEB references, and PSNR penalises that far more than SSIM."
    if clahe_p["d"] < 0 else
    f"we are {f2(clahe_p['d'])} dB higher in PSNR ({clahe_p['eff']} effect) and {sg(S['clahe_only']['ssim']['d'], 3)} in SSIM."
)

md = f"""# Presentation guide — Review II

How to present *Hybrid underwater image restoration* at the review. Written
for the three presenters. The deck is [Review_II_Presentation.pptx](Review_II_Presentation.pptx);
every slide's speaker notes contain a short version of the script below, so
PowerPoint's Presenter View shows it on the laptop screen.

---

## 1. Format and roles

**Assumed slot:** about 18 minutes of slides, 4 minutes of live demo, then
questions. If your slot is shorter, use the 15-minute version in Section 6.

| Part | Slides | Presenter | Time |
|---|---|---|---|
| Problem, goal and pipeline | 1–4 | **Kopal** | ~4 min |
| Method and evaluation protocol | 5–10 | **Yathaartha** | ~6 min |
| Results, finding, limitations and rigour | 11–16 | **Manav** | ~7 min |
| Live demo | 17 | **Manav** (Yathaartha drives the mouse if you prefer) | ~4 min |
| Takeaways and questions | 18 | **Kopal** opens Q&A; everyone answers | 1 min + Q&A |

This split is a suggestion: swap sections to match who knows each part best.
Everyone should be able to answer questions on any section (Section 5).

**The one-sentence story** (everyone should be able to say this):
> A classical pipeline that adapts each stage to the type of water, with every
> parameter tuned on 30% of UIEB and every claim tested on the other 70%. Its
> biggest gain comes from physically motivated red-channel compensation, and
> along the way we found that optimising popular no-reference quality scores
> actively hurts fidelity.

---

## 2. Before the review — checklist

**The day before**
- [ ] Rehearse once end to end with a timer, including the hand-offs.
- [ ] Open the deck in **Presenter View** (Slide Show → Use Presenter View) and check the notes appear.
- [ ] Every presenter skims Section 5 (questions).

**One hour before**
- [ ] Laptop **plugged in**, *Low Power Mode off*, notifications off (Focus → Do Not Disturb).
- [ ] The **external drive with UIEB is mounted** (`/Volumes/Lexar/raw-890`); the app needs it for dataset images and references.
- [ ] Start the app from the repo root: `./dip/bin/streamlit run app/streamlit_app.py`.
- [ ] In the app, open `{cmp_a.replace('.png', '')}` and `{weak}` once each so they are cached and switch instantly. Run the Downstream tab once.
- [ ] Browser zoom at 110–125% so the room can read it; close other tabs.
- [ ] Have `outputs/Review_II_Results_Section.docx` open in the background, in case a reviewer asks for detail.

**If anything fails during the demo:** say *"we'll show the same thing from
the recorded results"* and go to slides 5, 14 and 15. They show the same
stages and comparisons.

---

## 3. How to present well

- **Lead with the slide title.** Each title is the message ("Results: best SSIM, clearly ahead of HE and UDCP"). Say it first, then support it.
- **Point at the evidence.** On chart slides, name the bar you are talking about ("the teal bar is ours").
- **Be upfront about limitations.** Slide 15 is deliberate: reviewers trust a team that shows where its method loses. Never claim a win the numbers don't show.
- **Use the protocol as a strength.** "Tuned on 30%, reported on 70%" and "equally tuned ablation" answer most "is this fair?" questions before they're asked.
- **Hand-offs are one line.** "Yathaartha will now take you through each step." Then step back.
- **In Q&A, answer, then stop.** If you don't know, say what you would check ("we haven't measured that; we'd test it by …").

---

## 4. Slide-by-slide script

Times are targets. *Italic lines* are stage directions.

### Slide 1 · Title — Kopal (1 min)
"Good morning. We are presenting our Review II work: a hybrid underwater
image restoration pipeline. On the right is a real image from the UIEB
benchmark, before and after our method. In the next few minutes: why
underwater images look wrong, how our pipeline fixes each problem, how we
evaluated it fairly, and what we found, followed by a live demo."

### Slide 2 · Why underwater photos look wrong — Kopal (1 min)
"Underwater images degrade in three physical ways. *Point to each card.*
**Colour cast**: water absorbs red within the first few metres, so scenes
turn blue-green, like the image on the left. **Haze**: particles scatter
light back toward the camera, adding a fog that grows with distance. And
**low light and flat contrast** at depth. How strong each one is depends on
the water: deep water loses the most red, and turbid water scatters the
most. That observation drives our design."

### Slide 3 · Our goal and what is new — Kopal (1 min)
"Our goal: restore underwater images using only classical image processing,
with no deep learning, make every stage adapt to the water type, and prove
each gain on images never used for tuning. We contribute three novel
modules. Module 5 classifies the water and routes the parameters, including
how much red light to restore. Module 7 predicts, per image, how to blend
two complementary processing streams. Module 9 checks that restoration
actually helps computer vision. We also report a finding about popular
image-quality scores, which we'll come back to."

### Slide 4 · The pipeline at a glance — Kopal (1 min)
"Here is the whole pipeline. *Trace left to right.* The raw image is
classified by water type, which picks the parameters for every later stage.
Steps 1 to 3 fix colour and brightness. Then we create two versions of the
image: a detail stream with CLAHE and a haze-free stream with the underwater
dark channel prior. Module 7 decides how much to trust each, and a
multi-scale pyramid blends them. The teal boxes are our novel modules.
Yathaartha will now take you through each step."

### Slide 5 · Every stage on one held-out image — Yathaartha (1 min)
"This is one test image going through every stage. *Point along the top
row.* Red compensation warms the coral, white balance removes the blue cast,
and gamma adjusts brightness. *Bottom row.* The CLAHE stream has the most
texture but looks grey; the dehazed stream has depth and colour but is
darker. The fused result, framed, combines both: SSIM
{f3(ex['794_img_.png']['scores']['proposed_hybrid']['ssim'])} against
{f3(ex['794_img_.png']['scores']['clahe_only']['ssim'])} for CLAHE alone on this image."

### Slide 6 · Routing: four water types, four profiles — Yathaartha (1 min)
"Module 5 classifies every image with four cheap statistics, using these
rules in order. Each water type gets a profile of five parameters. These were
**not hand-tuned**: we calibrated them on 30% of the data, and a type keeps
its own values only if they beat a single global profile with statistical
significance. In the final calibration
{join([TL[t] for t in own_prof]) + ' kept their own stage-1 values' if own_prof else 'no type beat the global stage-1 values'},
and {join([TL[t] for t in own_red]) + ' kept its own red-compensation strength' if own_red else 'every type uses the global red strength'}."

### Slide 7 · Restoring the red light water absorbs — Yathaartha (1 min)
"Our biggest gain comes from this step. Water removes red so thoroughly that
scaling the red channel up just amplifies noise. Following Ancuti et al., we
rebuild red from the green channel, which survives better. The (1 − R) term
protects pixels that are already red. Calibration ranked the best strengths
{join([f"{TL[t]} ({N['red_types'][t]['best']:g})" for t in red_order])};
{('and ' + join([TL[t] for t in own_red]) + ' kept its own strength.') if own_red else f"none of these beat a single global strength significantly, so every image uses s = {N['red_global']:g}."}
**On held-out images this step alone adds {sg(R['psnr']['mean_diff'])} dB PSNR**
(95% CI {f2(R['psnr']['ci95_lo'])}–{f2(R['psnr']['ci95_hi'])}){', our largest single gain' if red_largest else ''}."

### Slide 8 · Correct the colour, then build two streams — Yathaartha (1 min)
"White balance makes the average colour neutral, with a cap so a nearly
empty channel isn't blown up. Gamma sets overall brightness{gamma_dir}. Then comes the key idea of a hybrid: two complementary streams.
CLAHE on the lightness channel recovers texture without shifting colour, but
looks grey on its own. The underwater dark channel prior estimates haze per
pixel, using only green and blue because red is always dark underwater, and
removes it. It restores depth but darkens. Neither is good alone."

### Slide 9 · Blending the streams per image — Yathaartha (1 min)
"Fusion uses weight maps for contrast, saturation and saliency, so each pixel
draws from whichever stream looks better there, and a Laplacian pyramid hides
the seams. On top of that, Module 7 predicts a global bias α for each image
from 26 cheap statistics. *Point at the chart.* If we could pick the perfect
α per image using the reference, PSNR would rise by
{f2(cv['oracle']['psnr'] - cv['fixed']['psnr'])} dB. Our predictor recovers
about {share}% of that in cross-validation, with no reference needed at run
time. The y-axis starts at {axis_min:g} dB to make the gap visible."

### Slide 10 · Evaluation: tune on 30%, report on 70% — Yathaartha (1 min)
"How we evaluated matters as much as what we built. Tuning on the images you
report on inflates results, so we split UIEB once: {N['n_cal']} images for
calibration and {N['n_test']} held out. Every parameter is chosen on the
first set; every number you'll see comes from the second. The ablation is
fair: 'routing off' uses a global profile tuned exactly the same way. And
with 600 images almost anything is 'significant', so we report Holm-corrected
tests with effect sizes and confidence intervals. One caveat: UIEB references
are the enhancements volunteers preferred, not ground truth. Manav will now
present the results."

### Slide 11 · Results — Manav (1.5 min)
"On the {N['n_test']} held-out images, the proposed hybrid reaches
**{f2(M['hybrid_full']['psnr'])} dB PSNR and {f3(M['hybrid_full']['ssim'])} SSIM**.
*Point at the teal bars.* It clearly beats histogram equalisation
({sg(S['conventional_he']['psnr']['d'])} dB) and the physics-based UDCP
({sg(S['udcp']['psnr']['d'])} dB). Against CLAHE-only, the strongest
classical baseline, {psnr_vs_clahe_line}"

### Slide 12 · What each part contributes — Manav (1 min)
"Breaking that down on held-out images: red-channel compensation
{sg(R['psnr']['mean_diff'])} dB; Module 7's predicted α
{sg(D['psnr']['fusion_optimizer_alone (row4-row3)'])} dB; per-type routing
{sg(D['psnr']['routing_alone (row5-row3)'])} dB. *Point at the orange bar.*
That is the original Module 7, which picked α by maximising image-quality
scores: {sg(-S['hybrid_nr_search']['psnr']['d'])} dB against our predictor.
That brings us to our main finding."

### Slide 13 · No-reference scores mislead the optimiser — Manav (1.5 min)
"Our first Module 7 did what many papers do: it chose the fusion weight that
maximised the image's own UCIQE and UIQM, with no reference needed. It
backfires. These scores reward extreme blends, so the search pushed α to 0
or 1 for **{boundary}% of images**.
{'Every score-maximising selector on this chart is worse than simply fixing α at 0.5. ' if worse_than_fixed else ''}The
oracle bar shows what is possible. On held-out images the search costs
{f2(S['hybrid_nr_search']['psnr']['d'])} dB against our predictor. HE tells
the same story: it tops UCIQE while losing on fidelity. **The lesson:
validate a quality score against references before you optimise it.**"

### Slide 14 · Visual comparison — Manav (1 min)
"Two held-out examples, chosen as typical for their water type (close to the
median gain over both CLAHE-only and HE) rather than cherry-picked. *Point at
the reference column, then at ours.* Top, green turbid water: PSNR
{f2(ex[cmp_a]['scores']['proposed_hybrid']['psnr'])} dB for ours against
{f2(ex[cmp_a]['scores']['clahe_only']['psnr'])} for CLAHE-only and
{f2(ex[cmp_a]['scores']['conventional_he']['psnr'])} for HE. Bottom, hazy bright
water: {f2(ex[cmp_b]['scores']['proposed_hybrid']['psnr'])} against
{f2(ex[cmp_b]['scores']['clahe_only']['psnr'])} and
{f2(ex[cmp_b]['scores']['conventional_he']['psnr'])}. Individual images vary,
which is why every claim we make is based on the {N['n_test']}-image averages."

### Slide 15 · Where it still falls short — Manav (1 min)
"We want to be upfront. First, **PSNR**: {('we are level with CLAHE-only overall, ' + sg(clahe_p['d']) + ' dB, which is not a meaningful difference') if psnr_level else (sg(clahe_p['d']) + ' dB against CLAHE-only overall')},
and CLAHE-only still scores higher on {share_below}% of test images, like this
{TL[N['weak_type']]} one. Look at the colours next to the reference. Second,
**routing**: there is about {f2(N['routing_oracle_gain'])} dB of per-image
headroom, and four discrete water types capture only
{sg(D['psnr']['routing_alone (row5-row3)'])} dB of it. Third,
{('**calibration chose the mildest values we searched** (' + join(edges) + '), so the references prefer gentler processing than our grid allowed.') if edges else '**calibration is coordinate-wise**, not one joint search.'}"

### Slide 16 · Rigour: issues we found and fixed — Manav (1 min)
"Before trusting any number, we audited the pipeline and the evaluation and
found silent bugs: nothing crashed, the numbers were just wrong. UCIQE was
computed on OpenCV's 8-bit Lab encoding, the original optimiser's objective
was almost entirely one term, the gamma formula was applied with the opposite
convention, and the first profiles had been judged on the same images they
were tuned on. Each fix is documented in the results report{gamma_tail}"

### Slide 17 · Live demo — Manav (4 min) → see Section 4a

### Slide 18 · Takeaways — Kopal (1 min)
"Three takeaways. *Read the three lines.* Thank you; we're happy to take
questions."

---

## 4a. Demo script (slide 17, ~4 minutes)

*Switch to the browser. The app should already be on `{cmp_a.replace('.png', '')}`.*

1. **Compare tab** (40 s): "This is a held-out green-turbid image. Here's raw against our result. Now switch the comparison to HE, then CLAHE-only, and compare each one's colours with ours." Point out whatever differs most on this image.
2. **Pipeline tab** (60 s): "Every stage, with the parameters this image received. Module 5 classified it as green turbid." Scroll to Module 7: "The predicted α is here. If I drag the slider to 0 the result is all dehazed stream, at 1 all CLAHE. The prediction sits in between."
3. **Toggle the modules** (40 s): in the sidebar, switch Module 7 off, then Module 5 off. "Watch PSNR and SSIM in the header: each module's contribution, live." Switch both back on.
4. **Metrics tab** (40 s): "The no-reference scores on the left favour HE; the PSNR/SSIM table against the reference tells the real story, which is our finding in one screen."
5. **Downstream tab** (30 s): click *Run downstream validation*. "More ORB keypoints and edges: restoration helps feature-based vision, not only looks."
6. **The limitation** (30 s): pick `{weak}` ({TL[N['weak_type']]}). "And here is where we're weaker: compare the colours with the reference."

*Switch back to the deck (slide 18).* Restoring one {rt['resolution'][1]}×{rt['resolution'][0]}
image takes about {rt['seconds_per_image']:.1f} s on this laptop, so every switch is near-instant.

---

## 5. Likely questions — prepared answers

**Why not deep learning?**
Course scope and interpretability: every step is a classical, explainable
operator. We position it honestly: learning-based UIEB methods such as
Water-Net report higher PSNR/SSIM, and we don't claim to beat them.

**Isn't calibrating on references just training?**
It's light supervision: 5 numbers per water type plus a 26-coefficient
linear model, chosen on 30% of the data and tested on the untouched 70%.
There is no network and nothing is learned per pixel. We say this explicitly
in the report.

**{'Why are you only level with CLAHE-only on PSNR?' if psnr_level else ('Why do you lose to CLAHE-only on PSNR?' if clahe_p['d'] < 0 else 'How big is the PSNR gain over CLAHE-only?')}**
{('CLAHE-only is a gentle, local operator that stays close to the moderate UIEB references. Our pipeline also dehazes and re-balances colour, which helps structure (SSIM ' + sg(S['clahe_only']['ssim']['d'], 3) + ') but can shift global brightness, which PSNR penalises heavily. CLAHE-only still wins PSNR on ' + str(share_below) + '% of images.') if psnr_level or clahe_p['d'] < 0 else ('The PSNR effect is ' + clahe_p['eff'] + '; the clearer gain is SSIM (' + sg(S['clahe_only']['ssim']['d'], 3) + ').')}

**How reliable are the UIEB references?**
They are volunteer-preferred enhancements chosen from 12 methods, not ground
truth. We report both reference-based and no-reference metrics, plus
downstream ones, and we discuss that caveat on slide 10.

**What exactly is new compared with Ancuti et al.'s red compensation?**
The compensation formula is theirs. What's ours is placing it before an
adaptive white balance in a two-stream fusion pipeline, searching its strength
per water type with a significance gate
({'kept per type for ' + join([TL[t] for t in own_red]) if own_red else 'no type beat one global strength, so we use s = ' + format(N['red_global'], 'g') + ' everywhere'}),
and measuring its held-out effect ({sg(R['psnr']['mean_diff'])} dB) in an equally tuned ablation.

**Why should we believe the routing results with only 15 hazy-bright calibration images?**
That is exactly why there is a significance gate: a small type keeps its own
values only if they beat the global profile at p < 0.05; otherwise it falls
back. All results are then measured on held-out images.

**How do you classify water type without labels?**
With four interpretable rules on image statistics (brightness, haze,
saturation, green dominance). There are no ground-truth water-type labels in
UIEB; the classifier is judged by whether routing helps on held-out images.

**Why Wilcoxon and not a t-test? What is rank-biserial r?**
Per-image metric differences aren't normally distributed, and Wilcoxon is
the standard paired non-parametric test. With {N['n_test']} pairs, p-values are tiny
even for trivial differences, so we report rank-biserial r as the effect size
(|r| < 0.1 negligible, < 0.3 small, < 0.5 medium, otherwise large) together
with Holm correction across all tests.

**What are UCIQE and UIQM, and why don't you trust them?**
They are popular no-reference underwater quality scores combining colour
spread, contrast and saturation (UCIQE) or colourfulness, sharpness and
contrast (UIQM). We still report them, but our α sweep shows they rank blends
differently from the references, and maximising them costs about
{f2(S['hybrid_nr_search']['psnr']['d'])} dB.

**How does the α predictor generalise?**
It was fitted with 5-fold cross-validation on the calibration split, and its
held-out effect is {sg(D['psnr']['fusion_optimizer_alone (row4-row3)'])} dB.
It is modest and honest about that: it recovers about {share}% of the
per-image headroom.

**How fast is it?**
About {rt['seconds_per_image']:.1f} s for a {rt['resolution'][1]}×{rt['resolution'][0]} image on this laptop (single image, CPU).

**What about video?**
Out of scope for this review; we focused on images and their evaluation.

**What would you do with more time?**
*(Only if asked.)* Close more of the per-image headroom, which is about
{f2(N['routing_oracle_gain'])} dB for routing and
{f2(cv['oracle']['psnr'] - cv['fixed']['psnr'])} dB for fusion, for example
with region-wise rather than global α.

---

## 6. 15-minute version

Skip slides **5, 8 and 16**: say one sentence for each and click past. Trim
the demo to steps 1, 3 and 6 (2 minutes). The speaking times become Kopal 4,
Yathaartha 3.5, Manav 5.5 and demo 2.

---

## 7. Key numbers to remember

| | |
|---|---|
| Split | {N['n_cal']} calibration / {N['n_test']} held-out test images |
| Proposed hybrid | {f2(M['hybrid_full']['psnr'])} dB PSNR · {f3(M['hybrid_full']['ssim'])} SSIM |
| vs CLAHE-only | {sg(S['clahe_only']['psnr']['d'])} dB · {sg(S['clahe_only']['ssim']['d'], 3)} SSIM |
| vs HE / UDCP | {sg(S['conventional_he']['psnr']['d'])} dB / {sg(S['udcp']['psnr']['d'])} dB |
| Red compensation alone | {sg(R['psnr']['mean_diff'])} dB · {sg(R['ssim']['mean_diff'], 3)} SSIM |
| Module 7 / Module 5 alone | {sg(D['psnr']['fusion_optimizer_alone (row4-row3)'])} dB / {sg(D['psnr']['routing_alone (row5-row3)'])} dB |
| Original Module 7 search | {sg(-S['hybrid_nr_search']['psnr']['d'])} dB vs. the predictor · α at 0/1 for {boundary}% of images |
| Calibration | {calib_line} |
"""
open(OUT, "w").write(md)
print("wrote", OUT)
