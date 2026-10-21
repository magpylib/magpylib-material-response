/*
Keep the magpylib-studio views in the page's theme.

A view is a page of its own, saved by magpylib-studio into the srcdoc of an
iframe of class "magpy-studio-frame" (see how_it_works.md), so it shares this
page's origin. It dresses itself from the colour it sits on, and looks again
whenever an attribute on its root or body changes. So: paint its body with this
page's background and text, and say the theme on its root, on load and on every
theme switch.
*/
(function () {
    "use strict";

    var SELECTOR = "iframe.magpy-studio-frame";

    function dress(frame) {
        var doc = frame.contentDocument;
        if (!doc || !doc.body) return;
        var page = window.getComputedStyle(document.body);
        doc.body.style.background = page.backgroundColor;
        doc.body.style.color = page.color;
        doc.documentElement.setAttribute(
            "data-theme",
            document.documentElement.dataset.theme || "light",
        );
    }

    function dressAll() {
        document.querySelectorAll(SELECTOR).forEach(dress);
    }

    function watch() {
        document.querySelectorAll(SELECTOR).forEach(function (frame) {
            frame.addEventListener("load", function () {
                dress(frame);
            });
            dress(frame);
        });
        new MutationObserver(dressAll).observe(document.documentElement, {
            attributes: true,
            attributeFilter: ["data-theme"],
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", watch);
    } else {
        watch();
    }
})();
