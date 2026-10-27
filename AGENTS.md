# NIF project instructions

## Tutorial and visualization fidelity

The original TensorFlow tutorial is the reference for the scientific case and
its presentation. Preserve its data definition, coordinate order, evaluation
grid, normalization, plotting method, contour levels, color limits and residual
definitions when comparing a PyTorch port.

- Check the original plotting cell before preparing a comparison. In particular,
  `contourf`, `pcolormesh` and `imshow` are not interchangeable presentations.
- If a different visualization is useful, keep an original-style comparison and
  clearly label the alternative method, units and sampling. Explain the change
  before drawing conclusions from its appearance.
- Distinguish original notebook outputs from figures redrawn from saved arrays.
  Distinguish display interpolation from additional model predictions, and
  training points from extra spatial or temporal evaluation points.
- Preserve existing notebooks, images and numerical reports. Put revised
  comparison figures in a separate version rather than overwriting the evidence.

## Coordinate verification before plotting

Verify the coordinate-to-value mapping point by point, not just array shapes:
input column names, flattening order, reshape dimensions, transpose, plotted
axes and physical/normalized units must agree. Use a non-square grid and an
asymmetric coordinate sentinel when testing rearrangement. Check a known
analytical field or source-data samples when available.

Do not add a transpose merely because an image looks unusual. First determine
whether the cause is coordinate order, coarse sampling, display interpolation,
color scaling or an actual prediction difference. Report what the evidence
establishes. Keep numerical accuracy and user acceptance separate from rendering
checks. When A/B validation is requested, A implements and B independently
validates; do not disclose B's private tests or expected results to A.

## Lesson retained from T7, 2026-10-27

The T7 comparison report changed the original `contourf` rendering to
`pcolormesh(shading="nearest")`. With only 10 time samples, this produced a
staircase appearance that suggested an X/T rearrangement problem. Independent
pointwise checks found the coordinate mapping correct: time-major data needs
`reshape(10, 200).T` to obtain the plotting matrix indexed by `[x, t]`.

The report was restored to the original-style contour rendering in a separate
version. Its predictions and absolute-error tables did not change. Remember
this as a rendering-method mismatch, not a confirmed X/T bug, and apply the
checks above to future tutorials and derived reports.
