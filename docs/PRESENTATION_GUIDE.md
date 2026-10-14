# Presentation guide — Review II

How to present *Hybrid underwater image restoration* at the review. Written
for the three presenters. The deck is [Review_II_Presentation.pptx](Review_II_Presentation.pptx);
every slide's speaker notes contain a short version of the script below, so
PowerPoint's Presenter View shows it on the laptop screen.

---

## 1. Format and roles

**Assumed slot:** about 16 minutes of slides, 4 minutes of live demo, then
questions. If your slot is shorter, use the 15-minute version in Section 6.

| Part | Slides | Presenter | Time |
|---|---|---|---|
| Problem, goal and pipeline | 1–4 | **Kopal** | ~4 min |
| Method and evaluation protocol | 5–10 | **Yathaartha** | ~6 min |
| Results, examples, limitations and rigour | 11–15 | **Manav** | ~5.5 min |
| Live demo | 16 | **Manav** (Yathaartha drives the mouse if you prefer) | ~4 min |
| Takeaways and questions | 17 | **Kopal** opens Q&A; everyone answers | 1 min + Q&A |

This split is a suggestion: swap sections to match who knows each part best.
Everyone should be able to answer questions on any section (Section 5).

**The one-sentence story** (everyone should be able to say this):
> A classical pipeline that adapts each stage to the type of water, with every
> parameter tuned on 30% of UIEB and every claim tested on the other 70%. Its
> biggest gain comes from red-channel compensation before white balance, and it
> has the best SSIM of every method we compared.

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
- [ ] In the app, open `773_img_` and `699_img_` once each so they are cached and switch instantly. Run the Downstream tab once.
- [ ] Browser zoom at 110–125% so the room can read it; close other tabs.
- [ ] Have `outputs/Review_II_Results_Section.docx` open in the background, in case a reviewer asks for detail.

**If anything fails during the demo:** say *"we'll show the same thing from
the recorded results"* and go to slides 5, 13 and 14. They show the same
stages and comparisons.

---

## 3. How to present well

- **Lead with the slide title.** Each title is the message ("Results: best SSIM, clearly ahead of HE and UDCP"). Say it first, then support it.
- **Point at the evidence.** On chart slides, name the bar you are talking about ("the teal bar is ours").
- **Be upfront about limitations.** Slide 14 is deliberate: reviewers trust a team that shows where its method loses. Never claim a win the numbers don't show.
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
actually helps computer vision."

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
0.875 against
0.831 for CLAHE alone on this image."

### Slide 6 · Routing: four water types, two profiles — Yathaartha (1 min)
"Module 5 classifies every image with four cheap statistics, using these
rules in order. Each water type gets a profile of five parameters. These were
**not hand-tuned**: we calibrated them on 30% of the data, and a type keeps
its own values only if they beat a single global profile with statistical
significance. In the final calibration
hazy bright kept their own stage-1 values,
and every type uses the global red strength."

### Slide 7 · Restoring the red light water absorbs — Yathaartha (1 min)
"Our biggest gain comes from this step. Water removes red so thoroughly that
scaling the red channel up just amplifies noise. Following Ancuti et al., we
rebuild red from the green channel, which survives better. The (1 − R) term
protects pixels that are already red. Calibration ranked the best strengths
clear blue (0.75), green turbid (0.75), deep low-light (0.5) and hazy bright (0);
none of these beat a single global strength significantly, so every image uses s = 0.75.
**On held-out images this step alone adds +0.52 dB PSNR**
(95% CI 0.41–0.63), our largest single gain."

### Slide 8 · Correct the colour, then build two streams — Yathaartha (1 min)
"White balance makes the average colour neutral, with a cap so a nearly
empty channel isn't blown up. Gamma sets overall brightness: values below 1 brighten. Then comes the key idea of a hybrid: two complementary streams.
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
0.64 dB. Our predictor recovers
about 17% of that in cross-validation, with no reference needed at run
time. The y-axis starts at 17 dB to make the gap visible."

### Slide 10 · Evaluation: tune on 30%, report on 70% — Yathaartha (1 min)
"How we evaluated matters as much as what we built. Tuning on the images you
report on inflates results, so we split UIEB once: 268 images for
calibration and 622 held out. Every parameter is chosen on the
first set; every number you'll see comes from the second. The ablation is
fair: 'routing off' uses a global profile tuned exactly the same way. And
with 600 images almost anything is 'significant', so we report Holm-corrected
tests with effect sizes and confidence intervals. One caveat: UIEB references
are the enhancements volunteers preferred, not ground truth. Manav will now
present the results."

### Slide 11 · Results — Manav (1.5 min)
"On the 622 held-out images, the proposed hybrid reaches
**18.27 dB PSNR and 0.871 SSIM**.
*Point at the teal bars.* It clearly beats histogram equalisation
(+1.91 dB) and the physics-based UDCP
(+5.29 dB). Against CLAHE-only, the strongest
classical baseline, we are level on PSNR (−0.04 dB, negligible, not significant) and +0.032 higher in SSIM (medium effect). *Point at the amber bar.* Against
the recipe practitioners usually bolt on, gray-world white balance plus CLAHE
with standard settings, we are +0.34 dB PSNR (small) and +0.033 SSIM (large)."

### Slide 12 · What each part contributes — Manav (1 min)
"Breaking that down on held-out images: red-channel compensation
+0.52 dB; Module 7's predicted α
+0.16 dB; per-type routing
+0.02 dB; and the two modules together
+0.18 dB on top of an equally tuned pipeline.
The compensation does most of the work; the modules add a smaller, measured
improvement."

### Slide 13 · Visual comparison — Manav (1 min)
"Two held-out examples, chosen as typical for their water type (close to the
median gain over both CLAHE-only and HE) rather than cherry-picked. *Point at
the reference column, then at ours.* Top, green turbid water: PSNR
21.59 dB for ours against
20.91 for CLAHE-only and
18.82 for HE. Bottom, hazy bright
water: 15.52 against
15.52 and
14.29. Individual images vary,
which is why every claim we make is based on the 622-image averages."

### Slide 14 · Where it still falls short — Manav (1 min)
"We want to be upfront. First, **PSNR**: we are level with CLAHE-only overall, −0.04 dB, which is not a meaningful difference,
and CLAHE-only still scores higher on 45% of test images, like this
clear blue one. Look at the colours next to the reference. Second,
**routing**: there is about 1.45 dB of per-image
headroom, and four discrete water types capture only
+0.02 dB of it. Third,
**calibration chose the mildest values we searched** (WB cap 1.5, CLAHE × 0.6 and ω 0.5), so the references prefer gentler processing than our grid allowed."

### Slide 15 · Rigour: issues we found and fixed — Manav (1 min)
"Before trusting any number, we audited the pipeline and the evaluation and
found silent bugs: nothing crashed, the numbers were just wrong. UCIQE was
computed on OpenCV's 8-bit Lab encoding, a 1,000-point ORB cap made raw and
restored keypoint counts tie, the gamma formula was applied with the opposite
convention, and the first profiles had been judged on the same images they
were tuned on. Each fix is documented in the results report, and after the last fix we re-ran the entire calibration and evaluation."

### Slide 16 · Live demo — Manav (4 min) → see Section 4a

### Slide 17 · Takeaways — Kopal (1 min)
"Three takeaways. *Read the three lines.* Thank you; we're happy to take
questions."

---

## 4a. Demo script (slide 16, ~4 minutes)

*Switch to the browser. The app should already be on `773_img_`.*

1. **Compare tab** (40 s): "This is a held-out green-turbid image. Here's raw against our result. Now switch the comparison to HE, then CLAHE-only, and compare each one's colours with ours." Point out whatever differs most on this image.
2. **Pipeline tab** (60 s): "Every stage, with the parameters this image received. Module 5 classified it as green turbid." Scroll to Module 7: "The predicted α is here. If I drag the slider to 0 the result is all dehazed stream, at 1 all CLAHE. The prediction sits in between."
3. **Toggle the modules** (40 s): in the sidebar, switch Module 7 off, then Module 5 off. "Watch PSNR and SSIM in the header: each module's contribution, live." Switch both back on.
4. **Metrics tab** (40 s): "The no-reference scores on the left favour HE; the PSNR/SSIM table against the reference tells the real story."
5. **Downstream tab** (30 s): click *Run downstream validation*. "More ORB keypoints and edges: restoration helps feature-based vision, not only looks."
6. **The limitation** (30 s): pick `699_img_` (clear blue). "And here is where we're weaker: compare the colours with the reference."

*Switch back to the deck (slide 17).* Restoring one 1024×718
image takes about 0.2 s on this laptop, so every switch is near-instant.

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

**Why are you only level with CLAHE-only on PSNR?**
CLAHE-only is a gentle, local operator that stays close to the moderate UIEB references. Our pipeline also dehazes and re-balances colour, which helps structure (SSIM +0.032) but can shift global brightness, which PSNR penalises heavily. CLAHE-only still wins PSNR on 45% of images.

**Isn't CLAHE what people normally use?**
In robotics and survey pipelines the usual quick fix is gray-world white
balance plus CLAHE, often straight from OpenCV, so we added exactly that as a
baseline, untuned with standard settings. The hybrid is +0.34 dB PSNR (small) and +0.033 SSIM (large).
Photographers mostly use Lightroom or apps like Dive+, which aren't
reproducible benchmarks; with depth data, survey teams use physics-based
methods such as Sea-thru, which UIEB can't support.

**How reliable are the UIEB references?**
They are volunteer-preferred enhancements chosen from 12 methods, not ground
truth. We report both reference-based and no-reference metrics, plus
downstream ones, and we discuss that caveat on slide 10.

**What exactly is new compared with Ancuti et al.'s red compensation?**
The compensation formula is theirs. What's ours is placing it before an
adaptive white balance in a two-stream fusion pipeline, searching its strength
per water type with a significance gate
(no type beat one global strength, so we use s = 0.75 everywhere),
and measuring its held-out effect (+0.52 dB) in an equally tuned ablation.

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
the standard paired non-parametric test. With 622 pairs, p-values are tiny
even for trivial differences, so we report rank-biserial r as the effect size
(|r| < 0.1 negligible, < 0.3 small, < 0.5 medium, otherwise large) together
with Holm correction across all tests.

**What are UCIQE and UIQM, and why don't you trust them?**
They are popular no-reference underwater quality scores combining colour
spread, contrast and saturation (UCIQE) or colourfulness, sharpness and
contrast (UIQM). We report them, but they reward over-enhancement: global HE
has the highest UCIQE and UIQM of all methods while losing clearly on PSNR and
SSIM. That's why our claims rest on the reference-based metrics.

**How does the α predictor generalise?**
It was fitted with 5-fold cross-validation on the calibration split, and its
held-out effect is +0.16 dB.
It is modest and honest about that: it recovers about 17% of the
per-image headroom.

**How fast is it?**
About 0.2 s for a 1024×718 image on this laptop (single image, CPU).

**What about video?**
Out of scope for this review; we focused on images and their evaluation.

**What would you do with more time?**
*(Only if asked.)* Close more of the per-image headroom, which is about
1.45 dB for routing and
0.64 dB for fusion, for example
with region-wise rather than global α.

---

## 6. 15-minute version

Skip slides **5, 8 and 15**: say one sentence for each and click past. Trim
the demo to steps 1, 3 and 6 (2 minutes). The speaking times become Kopal 4,
Yathaartha 3.5, Manav 5.5 and demo 2.

---

## 7. Key numbers to remember

| | |
|---|---|
| Split | 268 calibration / 622 held-out test images |
| Proposed hybrid | 18.27 dB PSNR · 0.871 SSIM |
| vs CLAHE-only | −0.04 dB · +0.032 SSIM |
| vs gray-world WB + CLAHE | +0.34 dB · +0.033 SSIM |
| vs HE / UDCP | +1.91 dB / +5.29 dB |
| Red compensation alone | +0.52 dB · +0.024 SSIM |
| Module 7 / Module 5 alone | +0.16 dB / +0.02 dB |
| Calibration | stages 1–2 alternated, converged after 1 round |
