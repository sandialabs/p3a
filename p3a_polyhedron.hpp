#pragma once

namespace p3a {

/**
 * \brief A polyhedron. Can be convex, nonconvex, even multiply-connected.
 *
 * \details The MaxVerts constant is more than just the maximum number
 *   of vertices of a representable polyhedron: the clip() code actually
 *   adds new vertices prior to removing old ones, so this actually
 *   needs to be the maximum number of representable vertices
 *   plus the maximum number of representable edges intersected by a plane
 */
template <class T, int MaxVerts = 128>
struct polyhedron {
  static constexpr int max_verts = MaxVerts;
  struct vertex {
    int pnbrs[3];    /*!< Neighbor indices. */
    position_quantity<T> pos; /*!< Vertex position. */
  };
  vertex verts[max_verts]; /*!< Vertex buffer. */
  int nverts;                 /*!< Number of vertices in the buffer. */
  [[nodiscard]] P3A_HOST P3A_DEVICE inline
  volume_quantity<T> volume() const;
  P3A_HOST P3A_DEVICE inline
  void clip(
      vector3<adimensional_quantity<T>> const& normal,
      length_quantity<T> const& distance);
};

/**
 * \brief Clip a polyhedron against an arbitrary number of clip planes (find its
 * intersection with a set of half-spaces).
 *
 * \param [in,out] poly
 * The polygon to be clipped.
 *
 * \param [in] planes
 * An array of planes against which to clip this polygon.
 *
 */
template <class T, int MaxVerts>
P3A_HOST P3A_DEVICE inline
void polyhedron<T, MaxVerts>::clip(
      vector3<adimensional_quantity<T>> const& normal,
      length_quantity<T> const& distance)
{
  if (this->nverts <= 0) return;
  // variable declarations
  int v, np, onv, vcur, vnext, numunclipped;
  // signed distances to the clipping plane
  length_quantity<T> sdists[MaxVerts];
  length_quantity<T> smin, smax;
  // calculate signed distances to the clip plane
  onv = this->nverts;
  smin = maximum_value<T>();
  smax = minimum_value<T>();
  // for marking clipped vertices
  int clipped[MaxVerts] = {};  // all initialized to zero
  for (v = 0; v < onv; ++v) {
    sdists[v] = distance + dot_product(this->verts[v].pos, normal);
    if (sdists[v] < smin) smin = sdists[v];
    if (sdists[v] > smax) smax = sdists[v];
    if (sdists[v] < zero_value<length_quantity<T>>()) clipped[v] = 1;
  }
  // skip this face if the poly lies entirely on one side of it
  if (smin >= zero_value<length_quantity<T>>()) return;
  if (smax <= zero_value<length_quantity<T>>()) {
    this->nverts = 0;
    return;
  }
  // check all edges and insert new vertices on the bisected edges
  for (vcur = 0; vcur < onv; ++vcur) {
    if (clipped[vcur]) continue;
    for (np = 0; np < 3; ++np) {
      vnext = this->verts[vcur].pnbrs[np];
      if (!clipped[vnext]) continue;
      this->verts[this->nverts].pnbrs[0] = vcur;
      this->verts[vcur].pnbrs[np] = this->nverts;
      this->verts[this->nverts].pos =
          (this->verts[vcur].pos * -sdists[vnext] + this->verts[vnext].pos * sdists[vcur]) /
          (-sdists[vnext] + sdists[vcur]);
      ++(this->nverts);
    }
  }
  // for each new vert, search around the poly for its new neighbors
  // and doubly-link everything
  for (auto vstart = onv; vstart < this->nverts; ++vstart) {
    auto vcur = vstart;
    auto vnext = this->verts[vcur].pnbrs[0];
    do {
      int np;
      for (np = 0; np < 3; ++np) {
        if (this->verts[vnext].pnbrs[np] == vcur) break;
      }
      vcur = vnext;
      auto pnext = (np + 1) % 3;
      vnext = this->verts[vcur].pnbrs[pnext];
    } while (vcur < onv);
    this->verts[vstart].pnbrs[2] = vcur;
    this->verts[vcur].pnbrs[1] = vstart;
  }
  // go through and compress the vertex list, removing clipped verts
  // and re-indexing accordingly (reusing `clipped` to re-index everything)
  numunclipped = 0;
  for (v = 0; v < this->nverts; ++v) {
    if (!clipped[v]) {
      this->verts[numunclipped] = this->verts[v];
      clipped[v] = numunclipped++;
    }
  }
  this->nverts = numunclipped;
  for (v = 0; v < this->nverts; ++v) {
    for (np = 0; np < 3; ++np) {
      this->verts[v].pnbrs[np] = clipped[this->verts[v].pnbrs[np]];
    }
  }
}

template <class T, int MaxVerts>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
volume_quantity<T> polyhedron<T, MaxVerts>::volume() const
{
  if (this->nverts <= 0) return zero_value<volume_quantity<T>>();
  volume_quantity<T> sixv = zero_value<volume_quantity<T>>();
  int np, vcur, vnext, pnext;
  // zero the moment
  // for keeping track of which edges have been visited
  int emarks[MaxVerts][3] = {{}};  // initialized to zero
  // loop over all vertices to find the starting point for each face
  for (int vstart = 0; vstart < this->nverts; ++vstart) {
    for (int pstart = 0; pstart < 3; ++pstart) {
      // skip this face if we have marked it
      if (emarks[vstart][pstart]) continue;
      // initialize face looping
      pnext = pstart;
      vcur = vstart;
      emarks[vcur][pnext] = 1;
      vnext = this->verts[vcur].pnbrs[pnext];
      position_quantity<T> const v0 = this->verts[vcur].pos;
      // move to the second edge
      for (np = 0; np < 3; ++np) {
        if (this->verts[vnext].pnbrs[np] == vcur) break;
      }
      vcur = vnext;
      pnext = (np + 1) % 3;
      emarks[vcur][pnext] = 1;
      vnext = this->verts[vcur].pnbrs[pnext];
      // make a triangle fan using edges
      // and first vertex
      while (vnext != vstart) {
        position_quantity<T> const v2 = this->verts[vcur].pos;
        position_quantity<T> const v1 = this->verts[vnext].pos;
        sixv += scalar_triple_product(v0, v1, v2);
        // move to the next edge
        for (np = 0; np < 3; ++np) {
          if (this->verts[vnext].pnbrs[np] == vcur) break;
        }
        vcur = vnext;
        pnext = (np + 1) % 3;
        emarks[vcur][pnext] = 1;
        vnext = this->verts[vcur].pnbrs[pnext];
      }
    }
  }
  return sixv / 6;
}

}
