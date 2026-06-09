/*
 * os-tabs.js - Auto-select and sync the OS-matching content tab.
 *
 * Material for MkDocs + pymdownx.tabbed (alternate_style: true) renders each
 * tab group as a set of CSS-driven radio inputs:
 *
 *   <div class="tabbed-set tabbed-alternate" data-tabs="N:3">
 *     <input id="__tabbed_N_1" name="__tabbed_N" type="radio" checked>
 *     <input id="__tabbed_N_2" name="__tabbed_N" type="radio">
 *     <input id="__tabbed_N_3" name="__tabbed_N" type="radio">
 *     <div class="tabbed-labels">
 *       <label for="__tabbed_N_1">Windows</label>
 *       <label for="__tabbed_N_2">macOS</label>
 *       <label for="__tabbed_N_3">Linux</label>
 *     </div>
 *     <div class="tabbed-content"> ... </div>
 *   </div>
 *
 * Which block is shown is pure CSS keyed on the checked radio, so selecting a
 * tab is just `input.checked = true` - no Material/content.tabs.link needed.
 *
 * Scope (important): this script ONLY touches tab groups whose label set is
 * EXACTLY {"Windows", "macOS", "Linux"} (case-sensitive). Every other OS-tabbed
 * page in the docs uses different labels ("MacOS", "Mac (Apple Silicon)",
 * "Linux (Fedora)", "macOS/Linux", ...), so none of them are affected. This is
 * why we do NOT enable the global `content.tabs.link` theme feature, which would
 * mis-link those pages.
 *
 * Behavior:
 *  - On load, select the visitor's detected OS in every canonical group.
 *  - A manual choice (clicking a canonical OS tab) is remembered in localStorage
 *    and wins over detection on later loads; it also syncs the page's other
 *    canonical groups so a Windows user only ever sees Windows commands.
 *  - Progressive enhancement: with JS off, pymdownx's default-checked first tab
 *    renders and everything still works.
 */
(function () {
  "use strict";

  // Exact, case-sensitive labels that opt a group into OS handling.
  var OS_BY_LABEL = { Windows: "windows", macOS: "macos", Linux: "linux" };
  var CANON = Object.keys(OS_BY_LABEL); // ["Windows", "macOS", "Linux"]
  var PREF_KEY = "sleap-os-tab";
  var BOUND_FLAG = "__sleapOsTabsBound";

  function loadPref() {
    try {
      return localStorage.getItem(PREF_KEY);
    } catch (e) {
      return null;
    }
  }

  function savePref(os) {
    try {
      localStorage.setItem(PREF_KEY, os);
    } catch (e) {
      /* storage unavailable (private mode / disabled) - ignore */
    }
  }

  // Detect OS -> "windows" | "macos" | "linux" | null.
  function detectOS() {
    var p = "";
    try {
      if (navigator.userAgentData && navigator.userAgentData.platform) {
        p = navigator.userAgentData.platform;
      }
    } catch (e) {
      /* ignore */
    }
    if (!p && navigator.platform) p = navigator.platform;
    var hay = (p + " " + (navigator.userAgent || "")).toLowerCase();

    if (hay.indexOf("win") !== -1) return "windows";
    // iOS / iPadOS / macOS all map to the macOS tab.
    if (
      hay.indexOf("mac") !== -1 ||
      hay.indexOf("iphone") !== -1 ||
      hay.indexOf("ipad") !== -1 ||
      hay.indexOf("ipod") !== -1
    ) {
      return "macos";
    }
    // Android is Linux-based -> Linux tab.
    if (hay.indexOf("linux") !== -1 || hay.indexOf("android") !== -1) {
      return "linux";
    }
    return null;
  }

  // Tab groups whose label set is EXACTLY {Windows, macOS, Linux}.
  function canonicalGroups() {
    var out = [];
    var sets = document.querySelectorAll(".tabbed-set");
    for (var i = 0; i < sets.length; i++) {
      var labels = sets[i].querySelectorAll(".tabbed-labels > label");
      if (labels.length !== CANON.length) continue;
      var texts = [];
      for (var j = 0; j < labels.length; j++) {
        texts.push((labels[j].textContent || "").trim());
      }
      var exact =
        CANON.every(function (c) {
          return texts.indexOf(c) !== -1;
        }) &&
        texts.every(function (t) {
          return CANON.indexOf(t) !== -1;
        });
      if (exact) out.push(sets[i]);
    }
    return out;
  }

  // Select the radio for `os` in a single group. Setting `.checked` flips the
  // CSS-driven display and, crucially, does NOT dispatch a change event, so the
  // sync handler below can't recurse.
  function selectInGroup(set, os) {
    var labels = set.querySelectorAll(".tabbed-labels > label");
    for (var i = 0; i < labels.length; i++) {
      if (OS_BY_LABEL[(labels[i].textContent || "").trim()] !== os) continue;
      var input = set.querySelector("#" + CSS.escape(labels[i].htmlFor));
      if (input) input.checked = true;
      return;
    }
  }

  function applyToAll(os, groups) {
    if (!os) return;
    for (var i = 0; i < groups.length; i++) selectInGroup(groups[i], os);
  }

  // User clicked an OS tab in a canonical group -> remember it and sync the rest.
  function onChange(event) {
    var input = event.target;
    if (!input || input.tagName !== "INPUT" || input.type !== "radio") return;
    var set = input.closest(".tabbed-set");
    if (!set) return;
    var groups = canonicalGroups();
    if (groups.indexOf(set) === -1) return;

    var label = set.querySelector('label[for="' + CSS.escape(input.id) + '"]');
    var os = label && OS_BY_LABEL[(label.textContent || "").trim()];
    if (!os) return;

    savePref(os);
    for (var i = 0; i < groups.length; i++) {
      if (groups[i] !== set) selectInGroup(groups[i], os);
    }
  }

  function run() {
    var groups = canonicalGroups();
    if (!groups.length) return; // nothing on this page opts in
    applyToAll(loadPref() || detectOS(), groups);
    // Delegated, capture-phase listener bound once for the document's lifetime
    // (survives Material instant navigation, which reuses the document).
    if (!document[BOUND_FLAG]) {
      document.addEventListener("change", onChange, true);
      document[BOUND_FLAG] = true;
    }
  }

  // Material exposes a global `document$` observable that emits on initial load
  // and after each instant navigation; prefer it, else fall back to DOM ready.
  function boot() {
    if (window.document$ && typeof window.document$.subscribe === "function") {
      window.document$.subscribe(function () {
        requestAnimationFrame(run);
      });
    } else if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () {
        requestAnimationFrame(run);
      });
    } else {
      requestAnimationFrame(run);
    }
  }

  boot();
})();
