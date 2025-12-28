// Theme management
const themeToggle = document.getElementById('themeToggle');
const html = document.documentElement;

// Load saved theme or system preference
const savedTheme = localStorage.getItem('theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
html.setAttribute('data-bs-theme', savedTheme);

themeToggle?.addEventListener('click', () => {
    const currentTheme = html.getAttribute('data-bs-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    html.setAttribute('data-bs-theme', newTheme);
    localStorage.setItem('theme', newTheme);
});

// Sidebar management
const sidebar = document.getElementById('sidebar');
const toggleBtn = document.getElementById('toggleSidebar');
const closeBtn = document.getElementById('closeSidebar');
const backdrop = document.getElementById('sidebarBackdrop');

function toggleSidebar() {
    sidebar.classList.toggle('show');
    backdrop.classList.toggle('show');
}

toggleBtn?.addEventListener('click', toggleSidebar);
closeBtn?.addEventListener('click', toggleSidebar);
backdrop?.addEventListener('click', toggleSidebar);

// Close sidebar on route change (mobile)
document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('.sidebar .nav-item');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth < 992) {
                toggleSidebar();
            }
        });
    });
});
