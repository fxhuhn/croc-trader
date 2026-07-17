/**
 * Interactive Accordion & Group Hover Handlers for Croc Trader Templates
 */

// Toggle Details Row Script (Desktop)
function toggleRow(accId) {
    const detailEl = document.getElementById(accId);
    if (!detailEl) return;

    const triggerEl = detailEl.previousElementSibling;
    const chevron = triggerEl ? triggerEl.querySelector('[data-lucide="chevron-down"]') : null;

    if (detailEl.classList.contains('hidden')) {
        detailEl.classList.remove('hidden');
        if (chevron) chevron.style.transform = 'rotate(180deg)';
    } else {
        detailEl.classList.add('hidden');
        if (chevron) chevron.style.transform = 'rotate(0deg)';
    }
}

// Toggle for grid-animated Position Accordions (Mobile)
function togglePositionAccordion(triggerButton, detailId) {
    const detailEl = document.getElementById(detailId);
    if (!detailEl) return;

    const isExpanded = triggerButton.getAttribute('aria-expanded') === 'true';
    const chevron = triggerButton.querySelector('.chevron-icon');

    if (isExpanded) {
        detailEl.style.gridTemplateRows = '0fr';
        triggerButton.setAttribute('aria-expanded', 'false');
        if (chevron) chevron.style.transform = 'rotate(0deg)';
    } else {
        detailEl.style.gridTemplateRows = '1fr';
        triggerButton.setAttribute('aria-expanded', 'true');
        if (chevron) chevron.style.transform = 'rotate(180deg)';
    }
}

// Group Highlights on Hover
document.addEventListener('DOMContentLoaded', () => {
    const linkables = document.querySelectorAll('.group-linkable');

    linkables.forEach(item => {
        const groupId = item.getAttribute('data-group-id');
        if (!groupId || groupId === '-' || groupId === 'None') return;

        item.addEventListener('mouseenter', () => {
            const matches = document.querySelectorAll(`.group-linkable[data-group-id="${groupId}"]`);
            matches.forEach(m => {
                m.classList.add('bg-indigo-50/20', 'shadow-sm');
                if (m.classList.contains('border-l-4')) {
                    m.classList.add('border-l-indigo-500');
                    m.classList.remove('border-l-transparent');
                }
            });
        });

        item.addEventListener('mouseleave', () => {
            const matches = document.querySelectorAll(`.group-linkable[data-group-id="${groupId}"]`);
            matches.forEach(m => {
                m.classList.remove('bg-indigo-50/20', 'shadow-sm');
                if (m.classList.contains('border-l-4')) {
                    m.classList.remove('border-l-indigo-500');
                    m.classList.add('border-l-transparent');
                }
            });
        });
    });
});
