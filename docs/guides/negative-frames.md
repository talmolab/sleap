# Negative Frames

*Case: Your model predicts animals on frames that are actually empty, and you
want to teach it that some frames contain no animals.*

A **negative frame** (also called a *background frame*) is a video frame that you
explicitly mark as containing **no animals**. When negative frames are included
in training, the model learns to predict *nothing* on empty frames, which
reduces false positives — spurious detections on background.

This is different from a frame that is simply *empty* (a labeled frame whose
instances were all deleted). An empty frame carries no information and is
discarded during training. A negative frame is a deliberate assertion that the
frame is background, and it *is* used as a training example.

!!! info "Why a deliberate action"
    Marking a frame negative is intentional by design. An accidental negative
    frame introduces a false negative into your training data, so SLEAP asks you
    to confirm before removing any existing instances, and gives you several ways
    to review which frames are marked (see [Reviewing negative
    frames](#reviewing-negative-frames)).

## Marking a frame as negative

Navigate to a frame that contains no animals, then mark it using any of:

- **Menu:** **Labels** → **Mark Frame as Negative**
- **Keyboard shortcut:** ++ctrl+m++
- **Right-click** in the video → **Mark Frame as Negative**

The menu item and context-menu entry are **checkable** — the checkmark shows
whether the current frame is currently marked as negative.

When the current frame already has instances on it, SLEAP shows a confirmation
dialog reporting how many instances will be removed. Marking the frame negative
removes those instances, because a frame with a labeled animal is not a
background frame. Adding a new instance to a negative frame automatically clears
the negative flag for the same reason.

To unmark a frame, use the same action again. If the frame is empty when you
unmark it, it is removed from the project entirely (an empty, non-negative frame
would otherwise be silently dropped during training).

## Reviewing negative frames

Negative frames are surfaced in a few places so accidental ones can be caught:

- **Seekbar marker:** every negative frame gets an amber tick on the seekbar.
  Hover over it to confirm ("negative (background) frame").
- **Status bar:** when the current frame is negative, the status bar shows a
  `[NEGATIVE FRAME]` tag.
- **Label Quality Control:** the [Label Quality
  Control](label-quality-control.md) checks flag any negative frame that still
  has instances — an inconsistency that usually means a mislabel.

## Training with negative frames

Negative frames are stored in your `.slp` project but are only used for training
when you opt in. In the **Training Pipeline** dialog, on the **Data** tab of a
model:

- **Use Negative Frames** — when enabled, every frame you marked as negative is
  added to the training set as a background example.
- **Negative Loss Weight** — the relative weight of the loss on negative frames.
  `1.0` weights them the same as labeled frames. Increase it above `1.0` if
  false positives persist, or decrease it to reduce their influence. It must be
  greater than `0`, and it is only used when *Use Negative Frames* is enabled.

!!! note "Supported model types"
    Negative frames are used by **single-instance**, **centroid**,
    **bottom-up**, and **multi-class bottom-up** models. They are ignored by the
    **centered-instance** and **multi-class top-down** heads, so the two options
    are hidden on those tabs. For a top-down pipeline, negative frames still help
    the **centroid** stage — which is the stage that produces the
    false-positive detections — so the options remain available there.

If you enable *Use Negative Frames* but have not marked any frames as negative,
SLEAP warns you in the training dialog, since the option would otherwise have no
effect.
