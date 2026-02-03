# Tracking Mistakes

Once you have predicted tracks, you'll need to proofread them to ensure instance identities are correct across frames.

!!! tip "Visualizing tracks"
    By default, predicted instances appear in grey and yellow. Select **"Color Predicted Instances"** to show tracks in color. Colors in the frame match colors in the seekbar and the **Instances** panel.

---

## Types of Tracking Errors

There are two main types of mistakes made by the tracking code:

1. **Mistaken Identities** — The tracker misidentifies which instance belongs to which track (two animals get "swapped")

2. **Lost Identities** — The tracker fails to link an instance to any previous track, creating a new track unexpectedly

---

## Fixing Mistaken Identities

When two instances are assigned to the wrong tracks, you need to **swap** their identities.

### Transpose Tracks

Use **Labels → Transpose Instance Tracks** to swap identities:

- If there are exactly 2 instances in the frame, they swap automatically
- If there are more, click the two instances you want to swap

![Fixing track identities](../assets/images/fixing-track.gif)

### Assign to Different Track

Use **Labels → Set Instance Track** to assign an instance to a different (or new) track.

!!! note "Propagation"
    When you assign an instance to a track, this change applies to all **subsequent frames**. For example, moving an instance from Track 3 to Track 2 will also move all Track 3 instances in later frames to Track 2—effectively "merging" the tracks.

---

## Fixing Lost Identities

When a new track is unexpectedly spawned, find where it started and reassign it to the correct track.

1. Use **Labels → Next Track Spawn Frame** to jump to the next frame where a new track begins
2. Select the instance and assign it to the correct track using **Labels → Set Instance Track**

---

## Quick Reference

| Action | How To |
|--------|--------|
| Select instance | Click in frame, click in Instances panel, or press `1`-`9` |
| View track name | Click an instance to see its track |
| Rename track | Double-click track name in Instances panel |
| Swap two tracks | **Labels → Transpose Instance Tracks** |
| Change track | **Labels → Set Instance Track** |
| Find new tracks | **Labels → Next Track Spawn Frame** |

---

For more tools and tips, see the [Proofreading guide](../guides/tracking-and-proofreading.md).
