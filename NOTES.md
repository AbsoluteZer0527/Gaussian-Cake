# Notes: Geometry Concepts in the Pipeline

Ideas from CSE 270 (discrete differential geometry) that the morph uses, grouped
by topic. Each row names where the code is and the identity or idea it rests on.

### Meshes and Topology

| Where | Concept | What it does |
|---|---|---|
| `MeshTopology::Analyze` | Euler–Poincaré, `χ = V − E + F` | Counts parts, boundary loops and the genus `g = (2·parts − χ − holes)/2`. Topology mode reports it, and `TopologyMismatch` warns when A and B differ (genus, boundary, orientability), since a shared mesh can't morph between them. |
| `MeshTopology::OrientConsistently` | an orientation (area form) exists | Fixes the triangle winding face by face, and detects non-orientable meshes (Möbius strip), which then use the SDF path. |
| `MeshTopology::OrientOutward` | divergence theorem | The signed volume `⅙ Σ v0 · (v1 × v2)` is positive exactly when the faces point outward, so a negative one flips the whole mesh. |
| `MeshRenderer::ComputeNormals`, `MorphTexturedRenderer` | area-weighted vertex normals | Each vertex normal is the sum of its faces' cross products (length = 2 × area), so big faces count more. |
| `MorphTexturedRenderer` (Topology) | point vs corner attributes | Uvs belong to corners (Houdini "vertex" attributes), not to points: two triangles meeting at a UV seam need different uvs at the same point, so every triangle keeps its own 3 corners. |

### Discrete Exterior Calculus

| Where | Concept | What it does |
|---|---|---|
| `MeshTopology::BuildForms` | `d1 ∘ d0 = 0` | Builds the discrete exterior derivatives (vertex → edge → face incidence). The test harness checks `\|d1 d0\| = 0`. |
| `MeshFitter` (Cotan Laplacian) | `Δ = ⋆d⋆d` → `L = d0ᵀ ⋆1 d0` | The template fit keeps its shape with the cotan Laplacian (the Hodge star `⋆1` is the cotan edge weights) instead of a uniform one. It respects the actual triangle shapes, and linear functions get exactly zero Laplacian. |
| `SplatHeat` cohesion (Splats) | heat equation `⋆0 ∂u/∂t = −d0ᵀ ⋆1 d0 u` | Smooths each splat's travel path over its neighbours with one implicit heat step on the splats' neighbour graph (edges = 1-cells, conductivities `⋆1`, splat masses `⋆0`), so neighbours move as chunks. The heat time sets the chunk size in distance, so it looks the same at 5k and 500k splats (repeated neighbour averaging made chunks shrink as the count grew). |
| `HeatWaveDelay` (Splats) | the same heat equation, from a source | Heat spreading from the lowest splats (the pan) decides when each splat leaves: the *Heat Wave* slider makes the morph spread up through the shape like a cake baking. Each splat runs its own clock, still exact at both ends. |

### Calculus on Manifolds (Pullbacks)

| Where | Concept | What it does |
|---|---|---|
| `sdf_common.glsl` `PhiGrad` | `d` commutes with pullback | The morph field is a pullback, `φ_t = (M_t⁻¹)*[(1−t)φ_A + t·T*φ_B]`, so the surface normal is exact: `∇φ_t = M_t⁻ᵀ[(1−t)∇φ_A + t·Aᵀ∇φ_B]`. Gradients are 1-forms and transform by the inverse transpose. This replaces the 6-sample finite difference. |
| `MidwayStretch` (Splats) | pullback of the area 2-form | A splat's patch of surface, spanned by tangents t₁, t₂, has area `\|M t₁ ∧ M t₂\|` under the OT path `M_t`. Splats grow or shrink mid-flight to match, so they keep covering the surface (toggle: *Area-preserving splats*). |
| Splat bake log | pullback of the volume form, `det J` | Logs `det J` of the OT map, the Monge–Ampère volume ratio: how much each region is compressed or expanded. |
| `MorphTexturedRenderer` (Topology) | pullback of a 0-form, `φ*c = c ∘ φ` | The template fit matches every template corner to a point on A and on B, so each corner carries A's and B's texture coordinates. The fragment shader blends the two real textures per pixel, so the detail no longer drops to blurry vertex colours when the blend starts. A triangle that lands across a seam or on two materials falls back to vertex colours on that side. |

### Sources

CSE 270 slides: 02 Mesh, 03 Half-Edge, 04 DEC, 05 Calculus on Manifolds, 06 Exterior Calculus Prelim, 07 Exterior Calculus.

Wang, Nabizadeh, Chern 2023, Exterior Calculus in Graphics (course notes).

Sorkine & Alexa 2007, As-Rigid-As-Possible Surface Modeling.
