# Notes

Time spent: ~10 hours.

## The idea

In my fourth year of university I designed a microcontroller case with AI help. The lid
was held on by tension against tabs sticking out from the sides of the base. The AI
kept getting it wrong. It put the tabs on the wrong side, and it didn't size the lid
properly relative to the case, so the two halves didn't fit together. The models looked
fine on screen. I only found out each time after a print.

That's the problem I wanted to work on here. Adam already does text-to-CAD, so I didn't
want to build another "type a prompt, get a part" demo. The harder problem is trusting
the output. A model will happily write a script that runs and gives you a valid solid
that's still wrong. The hole doesn't go all the way through, the fillet ends up on the
wrong edge, the boss is on the wrong face. It builds, so it looks fine.

A B-rep kernel can actually answer those questions exactly. Is this point inside
material? How thick is the wall along this line? What's the radius of this face? Is
this edge an inside or outside corner? So I built the agent around checking its own
work with the kernel:

write a script → build it → read the topology and look at renders → measure → fix → report

The output is the part plus a report saying what was checked and how.

Two decisions that ended up mattering most:

- **The script is the model.** The agent writes one short parametric Python script per
  version and the kernel rebuilds from scratch each time. The alternative, lots of small
  stateful tool calls, would mean ids going stale and the kernel's booleans eating their
  inputs. With scripts, versions are easy to diff and edit by hand, and if the kernel
  crashes I just replay the last script.
- **Same face/edge ids everywhere.** The topology listing, the labels on the rendered
  images, the selectors in the script and the 3D viewer all use the same ids (`F3`,
  `E12`). So when the model sees a face labelled F7 in an image it can look up exactly
  what F7 is.

## What I built vs. reused

Reused:
- `usd-brep` 1.0.1 from PyPI (SMLib kernel + OpenUSD 25.11 bundled). All the actual
  geometry: primitives, booleans, sweeps, lofts, fillets, shell, tessellation,
  closest-point queries, USD `BrepArray` import/export.
- three.js 0.160 for the browser viewer (vendored, MIT).
- I used Claude as a coding assistant throughout, which the brief allows. The idea,
  architecture, prompt design and the decisions below are mine.

Built:
- `kernel/dsl.py`: the scripting layer the model writes against. Wraps the kernel so
  nothing consumes its inputs, adds `at=`/`axis=` placement, extrude/revolve/loft/
  draft/sweep helpers, and edge/face selectors like `edges(body, convex=False,
  circular=True)`.
- `kernel/census.py`: lists every face and edge with type, size, position and
  convex/concave, using kernel queries. Also works on parts imported from USD, where
  every edge comes back as a spline, by fitting lines and circles to them.
- `kernel/mesh.py` and `kernel/render.py`: tessellation, mapping each triangle back to
  its face, and a small software renderer that draws 4 views with hidden lines removed
  and face ids written on them. Also section views.
- `kernel/worker.py`: runs the kernel in a separate process with timeouts, plus the
  measurement tools (point probe, distances, thickness along a ray).
- `kernel/assembly.py`: fit checks between parts. See the section below.
- `agent/`: the tool loop (`plan`, `build`, `look`, `measure`, `finish`), the prompt,
  and adapters for OpenAI (Responses API) and Anthropic.
- `server/`: FastAPI backend and a single-page UI.
- `tests/`: 20 tests, all runnable without an API key, including a scripted fake agent
  session where the "model" makes a blind-hole mistake and catches it.

## How I iterated on it

I tuned it against real runs on the bracket example in `examples/bracket.txt`.

**First run (GPT-4.1).** The part looked right: correct size, all holes through,
manifold. It even did a lot of checking, 13 measurements and a section. But it got two
things wrong. It put the "2 mm fillet where the boss meets the plate" on the top rim of
the boss instead of where the boss meets the plate. And it rounded the top edges of the
plate instead of the vertical corners. Then it marked everything as passing, I think
because there was a rounded face *somewhere* on the boss.

That was exactly the kind of mistake I wanted this to catch, so I changed a few things:

- After every build, the result now lists any sharp inside corners that are still there
  (from the topology listing). On the first run that would have shown the unfilleted
  edge at z = 8 right in front of the model.
- I spelled out in the prompt what "outer fillets" and "fillet at the junction" usually
  mean, and which selector picks each one.
- The prompt now says a fillet only counts if the sharp edge is gone *at that spot*, and
  that reporting a failure is better than a wrong pass.
- I added a `plan` step so the model writes down numbered requirements before building,
  and the final report has to go through them one by one.

**Second run (GPT-6.1 Sol).** Both fillets were right. It checked every hole with a ray,
cut a section through the boss, and worked out the expected volume by hand: 35,214 mm³
plus about 56 mm³ for the fillet, against 35,270 mm³ from the kernel (I checked the math).
All 7 requirements passed with evidence at the right places, in a single build.

To be fair, I switched to a stronger model and changed the prompt at the same time, so
two runs don't tell me which one fixed it. But the sharp-corner check doesn't depend on
the model, so that particular mistake would now get flagged either way.

The first run's mistake was the same kind as my case's tabs: a feature that exists
but in the wrong place. That's why proof in Foreman has to be positional. It's not
enough that a fillet or a tab exists somewhere; the measurement has to show it where
the brief says.

## Checking that parts fit together

After writing up the microcontroller case story I added the thing that would have
caught it. A script can now put several parts in their assembled positions
(`assembly = dict(base=base, lid=lid)`), and after every build the kernel reports, for
each pair of parts, whether they're separate, touching or overlapping, the smallest gap,
and the volume and bounding box of any overlap.

I tested it on a case with a lid held on by snap tabs, built three ways:

- Correct: 16.0 mm³ of overlap, located only at the two tabs (2 tabs × 10 × 2 mm ×
  0.4 mm, which matches).
- Tabs on the wrong side (my actual problem): zero overlap. The lid just rests on the rim
  and the tabs never engage.
- Lid sized wrong (my other problem): 501 mm³ of overlap along the whole wall, so it
  won't go on.

Those three cases are tests now. The build result spells out which situation it is, so
the model doesn't have to infer it from a picture. I also had to fix the renderer for
this. It used to sort whole triangles by depth, which broke when one part wraps around
another (the case wall was drawn on top of the lid), so I replaced it with a proper
per-pixel depth buffer.

This only checks the geometry in the assembled position. It doesn't simulate the tab
flexing as the lid goes on, so whether a 0.4 mm snap is too stiff for a given plastic is
still a judgment call.

## Other decisions

**Docker instead of a native install.** `usd-brep` only has wheels for Linux and Windows
on Python 3.12. There's no Mac wheel, and building from source on a Mac needs a separate
OpenUSD install. Docker on Apple Silicon uses the Linux ARM wheel, so `docker compose up`
works on a Mac with no extra setup.

**A small scripting layer instead of the raw kernel API.** The raw API is big and has
traps for a model: booleans destroy their inputs, cylinders only go along +Z, the basic
spline function needs at least 4 points and doesn't pass through them, and fillets need
edge handles from that exact body. The wrapper is about 30 functions and all of them are
safe to call in any order. The raw module is still available as `sm` if needed.

**Catching bad fillets.** The kernel will "succeed" on a 9 mm fillet on a 10 mm cube and
return a solid with *more* volume than the cube (1110 vs 1000 mm³) and four times the
surface area. So `fillet()` checks that the volume and area changed in a sane direction
and raises a clear error if not. The model gets told why, which turned out to be really
useful feedback.

**Working out convex vs. concave edges.** The stable Python API doesn't expose loop
orientation, so I take the outward normals of the two faces at the middle of the edge,
step a tiny bit in the direction of their difference, and ask the kernel if that point is
inside the solid. If it is, the edge is an inside corner.

**Software renderer.** It needed to run headless in the container and return images
right after each build. It's numpy + Pillow, about a second for 4 views including hidden
line removal. Face labels only go on faces that are actually visible, so the model isn't
reading labels for things behind other things. Interiors are checked with section views.

**Kernel in its own process.** Some operations can hang on bad input (there's an open
issue about a fillet that never returns). A separate process means I can kill it after a
timeout, and since every version is a script, recovering is just re-running it.

**One agent, not a planner + critic.** I considered splitting it up, but one model with
good tools was simpler. The renders, topology listing and measurements are what make it
correct itself, not extra agents. The function reference in the prompt is generated from
the code's docstrings so the two can't drift apart.

**OpenAI and Anthropic both work.** The loop uses one internal message format with an
adapter per provider. One gotcha: OpenAI's newer models (GPT-6 Astra, GPT-6.1 Sol) only
support tool calling through the Responses API, not Chat Completions. My first adapter
used Chat Completions, which worked for GPT-4.1, and I had to rewrite it.

## Limitations

- Shapes: boxes, cylinders, cones, spheres, tori, extrudes, revolves, lofts, drafted
  extrudes, sweeps, booleans, fillets/chamfers, shells, offsets and patterns. Not
  supported in the scripting layer: variable cross-section sweeps, sketch constraints,
  threads and mirror. The kernel has most of these, so they're reachable via `sm`.
- Surface types are classified by sampling normals, so unusual spline surfaces just show
  up as "freeform".
- Face/edge ids only hold for one build. The model re-reads them after each one.
- In an assembly, the detailed face/edge listing is only for the primary part. The
  other parts get the fit report, measurements and renders, but no face ids.
- Images are flat-shaded orthographic views. Fine for checking layout, but the actual
  numbers come from measurements, not from the model eyeballing the picture.
- One user per server session, nothing saved between restarts.
- A typical brief takes 1–5 builds and 1–4 minutes.
- I only did two live runs for tuning, so this isn't a real evaluation. The Anthropic
  adapter is tested offline but I didn't run it live.

## Assumptions

- Millimetres, Z up, parts sitting on Z = 0 unless the brief says otherwise.
- Through holes are cut with cylinders longer than the part, then checked with a ray.
- One primary part per session; extra parts that have to fit with it go in `assembly`.

## What I'd do next

1. Put dimensions on the rendered views (hole positions, overall size) so the model can
   read numbers off the drawing like a person would.
2. Better fit checks. The basic version is in (see above). Next would be per-face
   clearances (the gap on each wall, not just the smallest one), checking the lid can
   actually slide on along its path, and estimating snap tab strain for common print
   materials. Minimum wall thickness and draft angle checks would go in the same place.
3. Let the model patch parts of the script instead of rewriting it, and add sliders in
   the UI for the parameters at the top of the script.
4. Expand review mode (load an existing part and audit or modify it, which already works
   for single parts) to assemblies with multiple bodies.
5. More shapes: variable sweeps, mirror, threads.

## Problems I hit with the kernel package

- `PolyBrep.ray_intersection` failed with `Could not activate keep_alive!` on a mesh
  created inside a function, so I wrote my own ray casting in numpy.
- Exporting to USD prints a bunch of harmless `pxr` warnings, because the wheel bundles
  OpenUSD without its Python package.
- Fillets that are too large silently return a wrong solid (see above).
