/* Interaksi bersama: tema terang/gelap, menu mobile, sidebar admin, notifikasi. */
(function () {
    'use strict';

    const THEME_KEY = 'nilai-tanah-theme';
    const root = document.documentElement;

    function currentTheme() {
        if (root.dataset.theme) return root.dataset.theme;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
        const sync = () => btn.setAttribute('aria-label', currentTheme() === 'dark' ? 'Ganti ke mode terang' : 'Ganti ke mode gelap');
        sync();
        btn.addEventListener('click', () => {
            const next = currentTheme() === 'dark' ? 'light' : 'dark';
            root.dataset.theme = next;
            try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* penyimpanan tidak tersedia */ }
            sync();
        });
    });

    // Menu navigasi publik (mobile)
    const navToggle = document.querySelector('[data-nav-toggle]');
    const siteNav = document.getElementById('site-nav');
    if (navToggle && siteNav) {
        navToggle.addEventListener('click', () => {
            const open = siteNav.classList.toggle('open');
            navToggle.setAttribute('aria-expanded', open);
        });
        siteNav.querySelectorAll('a').forEach(a => a.addEventListener('click', () => {
            siteNav.classList.remove('open');
            navToggle.setAttribute('aria-expanded', 'false');
        }));
    }

    // Sidebar admin (mobile/tablet)
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.side-overlay');
    const sidebarToggle = document.querySelector('[data-sidebar-toggle]');
    function setSidebar(open) {
        if (!sidebar) return;
        sidebar.classList.toggle('open', open);
        if (overlay) overlay.classList.toggle('open', open);
        if (sidebarToggle) sidebarToggle.setAttribute('aria-expanded', open);
    }
    if (sidebarToggle) sidebarToggle.addEventListener('click', () => setSidebar(!sidebar.classList.contains('open')));
    if (overlay) overlay.addEventListener('click', () => setSidebar(false));
    document.addEventListener('keydown', e => { if (e.key === 'Escape') setSidebar(false); });

    // Tutup notifikasi
    document.querySelectorAll('.alert-close').forEach(btn => {
        btn.addEventListener('click', () => btn.closest('.alert').remove());
    });
})();
