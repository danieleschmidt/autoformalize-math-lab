// Custom JavaScript for autoformalize-math-lab documentation

document.addEventListener('DOMContentLoaded', function() {
    
    // Add copy button functionality for code blocks
    addCopyButtons();
    
    // Add proof system indicators
    addProofSystemIndicators();
    
    // Add mathematical notation enhancement
    enhanceMathematicalNotation();
    
    // Add interactive examples
    addInteractiveExamples();
    
    // Add benchmark visualizations
    addBenchmarkVisualizations();
});

function addCopyButtons() {
    // Enhanced copy button functionality for different code types
    const codeBlocks = document.querySelectorAll('div.highlight');
    
    codeBlocks.forEach(function(block) {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.title = 'Copy to clipboard';
        
        button.addEventListener('click', function() {
            const code = block.querySelector('pre').textContent;
            navigator.clipboard.writeText(code).then(function() {
                button.textContent = 'Copied!';
                button.classList.add('copied');
                setTimeout(function() {
                    button.textContent = 'Copy';
                    button.classList.remove('copied');
                }, 2000);
            });
        });
        
        block.appendChild(button);
    });
}

function addProofSystemIndicators() {
    // Add indicators for different proof systems
    const codeBlocks = document.querySelectorAll('div.highlight');
    
    codeBlocks.forEach(function(block) {
        const pre = block.querySelector('pre');
        if (!pre) return;
        
        const code = pre.textContent;
        let system = '';
        
        if (code.includes('theorem') && code.includes(':=')) {
            system = 'lean';
        } else if (code.includes('theorem') && code.includes('proof')) {
            system = 'isabelle';
        } else if (code.includes('Theorem') && code.includes('Qed.')) {
            system = 'coq';
        }
        
        if (system) {
            const indicator = document.createElement('span');
            indicator.className = `proof-system-indicator ${system}`;
            indicator.textContent = system.charAt(0).toUpperCase() + system.slice(1);
            block.insertBefore(indicator, pre);
        }
    });
}

function enhanceMathematicalNotation() {
    // Add hover tooltips for mathematical concepts
    const mathElements = document.querySelectorAll('.math, .MathJax');
    
    mathElements.forEach(function(element) {
        element.addEventListener('mouseenter', function() {
            // Could add tooltips for mathematical definitions
        });
    });
}

function addInteractiveExamples() {
    // Add interactive features to code examples
    const examples = document.querySelectorAll('.code-example');
    
    examples.forEach(function(example) {
        const title = example.querySelector('.code-title');
        const content = example.querySelector('.highlight');
        
        if (title && content) {
            title.style.cursor = 'pointer';
            title.addEventListener('click', function() {
                if (content.style.display === 'none') {
                    content.style.display = 'block';
                    title.textContent = title.textContent.replace('▶', '▼');
                } else {
                    content.style.display = 'none';
                    title.textContent = title.textContent.replace('▼', '▶');
                }
            });
        }
    });
}

function addBenchmarkVisualizations() {
    // Add visual enhancements to benchmark data
    const successRates = document.querySelectorAll('[data-success-rate]');
    
    successRates.forEach(function(element) {
        const rate = parseFloat(element.dataset.successRate);
        let className = 'low';
        
        if (rate >= 80) className = 'high';
        else if (rate >= 60) className = 'medium';
        
        element.classList.add('success-rate', className);
    });
}

// Add smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(function(link) {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add search enhancement
function enhanceSearch() {
    const searchInput = document.querySelector('input[name="q"]');
    if (searchInput) {
        searchInput.addEventListener('keyup', function(e) {
            if (e.key === 'Enter') {
                // Could add enhanced search functionality
            }
        });
    }
}

// Add theme toggle (if needed)
function addThemeToggle() {
    const toggle = document.createElement('button');
    toggle.id = 'theme-toggle';
    toggle.textContent = '🌓';
    toggle.title = 'Toggle dark/light theme';
    toggle.style.position = 'fixed';
    toggle.style.top = '10px';
    toggle.style.right = '10px';
    toggle.style.zIndex = '1000';
    toggle.style.background = 'none';
    toggle.style.border = 'none';
    toggle.style.fontSize = '20px';
    toggle.style.cursor = 'pointer';
    
    toggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-theme');
        localStorage.setItem('theme', 
            document.body.classList.contains('dark-theme') ? 'dark' : 'light'
        );
    });
    
    // Restore saved theme
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-theme');
    }
    
    document.body.appendChild(toggle);
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+K or Cmd+K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[name="q"]');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Esc to clear search
    if (e.key === 'Escape') {
        const searchInput = document.querySelector('input[name="q"]');
        if (searchInput && document.activeElement === searchInput) {
            searchInput.value = '';
            searchInput.blur();
        }
    }
});

// Add performance monitoring
function trackPerformance() {
    // Track page load performance
    window.addEventListener('load', function() {
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        console.log(`Documentation page loaded in ${loadTime}ms`);
    });
}

// Initialize additional features
trackPerformance();
enhanceSearch();