// ========================================
// KSAMDS Enhanced UI - JavaScript
// ========================================

document.addEventListener('DOMContentLoaded', () => {
  
  // ========== 1. Keyboard Shortcut for Search ==========
  const searchInput = document.querySelector('#search');
  window.addEventListener('keydown', (e) => {
    const isFormElement = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
    if (e.key === '/' && !isFormElement) {
      e.preventDefault();
      searchInput.focus();
    }
  });

  // ========== 2. Close Other Dropdowns When One Opens ==========
  document.querySelectorAll('.dd > summary').forEach(summary => {
    summary.addEventListener('click', (e) => {
      const currentDetails = summary.closest('.dd');
      const isCurrentlyOpen = currentDetails.hasAttribute('open');
      
      // Close all other dropdowns
      if (!isCurrentlyOpen) {
        document.querySelectorAll('.dd[open]').forEach(openDetails => {
          if (openDetails !== currentDetails) {
            openDetails.removeAttribute('open');
          }
        });
      }
    });
  });

  // ========== 3. Update Dimension Counts ==========
  function updateDimensionCounts(panel) {
    panel.querySelectorAll('.dimension-group').forEach(group => {
      const count = group.querySelectorAll('input[type="checkbox"]:checked').length;
      const badge = group.querySelector('.dimension-header__count');
      if (badge) {
        badge.textContent = count;
        badge.style.display = count > 0 ? 'inline-flex' : 'inline-flex';
        badge.style.background = count > 0 ? '#2563eb' : '#dbeafe';
        badge.style.color = count > 0 ? '#fff' : '#1e40af';
      }
    });
  }

  // ========== 4. Update Total Selection Count ==========
  function updateTotalCount(panel) {
    const total = panel.querySelectorAll('input[type="checkbox"]:checked').length;
    const totalSpan = panel.querySelector('.total-selected');
    if (totalSpan) {
      totalSpan.textContent = total;
    }
  }

  // ========== 5. Update Pill Badge ==========
  function updatePillBadge(details) {
    const panel = details.querySelector('.panel');
    const badge = details.querySelector('.pill .badge');
    const total = panel.querySelectorAll('input[type="checkbox"]:checked').length;
    
    if (badge) {
      badge.textContent = total;
      badge.style.display = total > 0 ? 'inline-flex' : 'none';
    }
  }

  // ========== 6. Listen to Checkbox Changes ==========
  document.querySelectorAll('.dd').forEach(details => {
    const panel = details.querySelector('.panel');
    
    panel.addEventListener('change', (e) => {
      if (e.target.type === 'checkbox') {
        updateDimensionCounts(panel);
        updateTotalCount(panel);
        updatePillBadge(details);
      }
    });
  });

  // ========== 7. Clear Button ==========
  document.querySelectorAll('.clear-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const panel = btn.closest('.panel');
      const details = btn.closest('.dd');
      
      panel.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
        cb.checked = false;
      });
      
      updateDimensionCounts(panel);
      updateTotalCount(panel);
      updatePillBadge(details);
    });
  });

  // ========== 8. Apply Filters Button ==========
  document.querySelectorAll('.apply-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const details = btn.closest('.dd');
      const panel = details.querySelector('.panel');
      const filterType = details.dataset.filter;
      
      // Collect selected filters by dimension
      const selectedFilters = {};
      panel.querySelectorAll('.dimension-group').forEach(group => {
        const dimension = group.dataset.dimension;
        const selected = Array.from(group.querySelectorAll('input[type="checkbox"]:checked'))
          .map(cb => cb.value);
        
        if (selected.length > 0) {
          selectedFilters[dimension] = selected;
        }
      });
      
      console.log(`${filterType} filters:`, selectedFilters);
      
      // Call API with selectedFilters
      await fetchResults(filterType, selectedFilters);
      
      // Close the dropdown
      details.removeAttribute('open');
      
      // Show notification
      showNotification(`Applied ${Object.keys(selectedFilters).length} dimension filter(s) for ${filterType}`);
    });
  });

  // ========== 9. Search Within Dropdown ==========
  document.querySelectorAll('.dimension-search').forEach(input => {
    input.addEventListener('input', (e) => {
      const searchTerm = e.target.value.toLowerCase();
      const panel = input.closest('.panel');
      
      panel.querySelectorAll('.dimension-group').forEach(group => {
        let hasVisibleItems = false;
        
        group.querySelectorAll('.dimension-items li').forEach(li => {
          const text = li.textContent.toLowerCase();
          const matches = text.includes(searchTerm);
          li.style.display = matches ? 'block' : 'none';
          if (matches) hasVisibleItems = true;
        });
        
        // Hide entire dimension group if no matches
        group.style.display = hasVisibleItems || searchTerm === '' ? 'block' : 'none';
      });
    });
  });

  // ========== 10. Main Search Button ==========
  document.querySelector('.icon-btn').addEventListener('click', async () => {
    const searchTerm = searchInput.value.trim();
    
    // Collect all active filters from all dropdowns
    const allFilters = {};
    document.querySelectorAll('.dd').forEach(details => {
      const filterType = details.dataset.filter;
      const panel = details.querySelector('.panel');
      const filters = {};
      
      panel.querySelectorAll('.dimension-group').forEach(group => {
        const dimension = group.dataset.dimension;
        const selected = Array.from(group.querySelectorAll('input[type="checkbox"]:checked'))
          .map(cb => cb.value);
        
        if (selected.length > 0) {
          filters[dimension] = selected;
        }
      });
      
      if (Object.keys(filters).length > 0) {
        allFilters[filterType] = filters;
      }
    });
    
    console.log('Main Search Query:', {
      searchTerm,
      filters: allFilters
    });
    
    // Call main search API
    await performSearch(searchTerm, allFilters);
    
    const filterCount = Object.keys(allFilters).length;
    const message = searchTerm 
      ? `Searching for "${searchTerm}" with ${filterCount} filter group(s)`
      : `Searching with ${filterCount} filter group(s)`;
    
    showNotification(message);
  });

  // ========== 11. Initialize Counts on Load ==========
  document.querySelectorAll('.dd').forEach(details => {
    const panel = details.querySelector('.panel');
    updateDimensionCounts(panel);
    updateTotalCount(panel);
    updatePillBadge(details);
  });

  // ========== 12. Close Dropdown on Outside Click ==========
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.dd')) {
      document.querySelectorAll('.dd[open]').forEach(details => {
        details.removeAttribute('open');
      });
    }
  });

  console.log('✅ KSAMDS Enhanced UI initialized with dimensional filters');
  
  // Initialize autocomplete and relationship modal
  initAutocomplete();
  initRelationshipModal();
});


// ========================================
// API INTEGRATION HELPERS
// ========================================

const API_BASE_URL = "http://localhost:8000";

/**
 * Fetch dimension data from backend API
 * Populate dimension dropdowns with real data from database
 */
async function loadDimensions() {
  try {
    console.log('🔄 Loading dimensions from API...');
    
    const response = await fetch(`${API_BASE_URL}/api/filters/all`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    console.log('✅ Dimensions loaded from API:', data);
    
    // Populate each entity type's dimensions
    for (const [entityType, filterData] of Object.entries(data)) {
      console.log(`📝 Populating ${entityType} dimensions...`);
      populateDimensionDropdown(entityType, filterData);
    }
    
    console.log('✅ All dimensions populated successfully');
    
  } catch (error) {
    console.error('❌ Failed to load dimensions:', error);
    showNotification('Failed to load filter options. Please refresh the page.', 'error');
  }
}

/**
 * Populate dimension dropdown with API data
 * @param {string} entityType - Entity type (knowledge, skills, abilities, etc.)
 * @param {Object} filterData - Filter data from API
 */
function populateDimensionDropdown(entityType, filterData) {
  const dropdown = document.querySelector(`.dd[data-filter="${entityType}"]`);
  if (!dropdown) return;
  
  const panel = dropdown.querySelector('.panel__dimensions');
  if (!panel) return;
  
  // Clear existing dimensions (except headers)
  panel.innerHTML = '';
  
  // Add each dimension group
  filterData.dimensions.forEach(dimensionGroup => {
    const groupHtml = `
      <div class="dimension-group" data-dimension="${dimensionGroup.dimension}">
        <div class="dimension-header">
          <div class="dimension-header__title">
            <span>${dimensionGroup.label}</span>
            <span class="dimension-header__count">0</span>
          </div>
          <svg class="dimension-header__icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>
        <ul class="dimension-items">
          ${dimensionGroup.options.map(option => `
            <li>
              <label>
                <input type="checkbox" value="${option.value}">
                <span class="item-name">${option.label}</span>
              </label>
            </li>
          `).join('')}
        </ul>
      </div>
    `;
    panel.insertAdjacentHTML('beforeend', groupHtml);
  });
  
  // Add dimension header click handlers
  panel.querySelectorAll('.dimension-header').forEach(header => {
    header.addEventListener('click', () => {
      const group = header.closest('.dimension-group');
      group.classList.toggle('collapsed');
    });
  });
}

/**
 * Perform search with selected filters for a specific entity type
 * @param {string} entityType - Entity type (knowledge, skills, abilities, functions, tasks)
 * @param {Object} selectedFilters - Selected dimensional filters
 * @returns {Promise<Object>} Search results
 */
async function fetchResults(entityType, selectedFilters) {
  try {
    // GET THE SEARCH TERM FROM INPUT
    const searchTerm = document.querySelector('#search').value.trim();
    
    const response = await fetch(`${API_BASE_URL}/api/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: searchTerm || undefined,
        entity_type: entityType,
        filters: selectedFilters,
        limit: 50,
        offset: 0
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const results = await response.json();
    
    console.log(`${entityType} results:`, results);
    
    // Display occupation info
    if (results.occupation) {
      displayOccupationInfo(results.occupation);
    }
    
    // Display results in the appropriate content box
    displayResults(`${entityType}-box`, results);
    
    return results;
    
  } catch (error) {
    console.error('Search failed:', error);
    showNotification(`Search failed for ${entityType}`, 'error');
    return null;
  }
}

/**
 * Perform main search across all entities
 * @param {string} searchTerm - Main search term
 * @param {Object} allFilters - All dimensional filters from all dropdowns
 * @returns {Promise<Object>} Search results for all entity types
 */
async function performSearch(searchTerm, allFilters) {
  try {
    // Search each entity type separately
    const entityTypes = ['knowledge', 'skills', 'abilities', 'functions', 'tasks'];
    const searchPromises = entityTypes.map(async (entityType) => {
      const filters = allFilters[entityType] || {};
      
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchTerm || undefined,
          entity_type: entityType,
          filters: filters,
          limit: 50,
          offset: 0
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return { entityType, data: await response.json() };
    });
    
    const results = await Promise.all(searchPromises);
    
    // Display occupation info (from first result)
    if (results.length > 0 && results[0].data.occupation) {
      displayOccupationInfo(results[0].data.occupation);
    }
    
    // Display results for each entity type
    results.forEach(({ entityType, data }) => {
      displayResults(`${entityType}-box`, data);
    });
    
    console.log('Search results:', results);
    
    return results;
    
  } catch (error) {
    console.error('Main search failed:', error);
    showNotification('Search failed', 'error');
    return null;
  }
}

/**
 * Load more results for an entity type
 * @param {string} entityType - Entity type to load more results for
 */
async function loadMoreResults(entityType) {
  const box = document.getElementById(`${entityType}-box`);
  const currentItems = box.querySelectorAll('.result-item').length;
  
  const searchTerm = document.querySelector('#search').value.trim();
  
  // Get active filters for this entity type
  const filterDropdown = document.querySelector(`.dd[data-filter="${entityType}"]`);
  const panel = filterDropdown.querySelector('.panel');
  const filters = {};
  
  panel.querySelectorAll('.dimension-group').forEach(group => {
    const dimension = group.dataset.dimension;
    const selected = Array.from(group.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);
    
    if (selected.length > 0) {
      filters[dimension] = selected;
    }
  });
  
  try {
    // Show loading state on button
    const loadMoreBtn = box.querySelector('.load-more-btn');
    if (loadMoreBtn) {
      loadMoreBtn.textContent = 'Loading...';
      loadMoreBtn.disabled = true;
    }
    
    // Fetch next batch
    const response = await fetch(`${API_BASE_URL}/api/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: searchTerm || undefined,
        entity_type: entityType,
        filters: filters,
        limit: 50,
        offset: currentItems  // Start from where we left off
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Append new results
    appendResults(`${entityType}-box`, data);
    
    showNotification(`Loaded ${data.results.length} more ${entityType}`);
    
  } catch (error) {
    console.error(`Failed to load more ${entityType}:`, error);
    showNotification(`Failed to load more ${entityType}`, 'error');
    
    // Re-enable button on error
    const loadMoreBtn = box.querySelector('.load-more-btn');
    if (loadMoreBtn) {
      loadMoreBtn.textContent = 'Load More';
      loadMoreBtn.disabled = false;
    }
  }
}

/**
 * Append results to existing content (for load more)
 * @param {string} boxId - ID of content box
 * @param {Object} data - API response data with results and has_more
 */
function appendResults(boxId, data) {
  const box = document.getElementById(boxId);
  if (!box) return;
  
  const entityType = boxId.replace('-box', '');
  
  // Remove "Load More" button if it exists
  const existingBtn = box.querySelector('.load-more-btn');
  if (existingBtn) existingBtn.remove();
  
  // Add new items
  data.results.forEach(item => {
    const itemEl = document.createElement('div');
    itemEl.className = 'result-item';
    itemEl.dataset.id = item.id;
    itemEl.dataset.name = item.name;
    itemEl.dataset.type = entityType;
    
    itemEl.innerHTML = `
      <div class="result-item__header">
        <h3 class="result-item__title">${item.name}</h3>
      </div>
    `;
    
    itemEl.addEventListener('click', () => {
      openRelationshipModal(item.id, item.name, entityType);
    });
    
    box.appendChild(itemEl);
  });
  
  // Add "Load More" button if there are more results
  if (data.has_more) {
    const loadMoreBtn = document.createElement('button');
    loadMoreBtn.className = 'load-more-btn';
    loadMoreBtn.textContent = 'Load More';
    loadMoreBtn.onclick = () => loadMoreResults(entityType);
    box.appendChild(loadMoreBtn);
  }
  
  // Update count badge
  const totalItems = box.querySelectorAll('.result-item').length;
  updateCountBadge(boxId, totalItems);
}

/**
 * Display occupation information
 * @param {Object} occupation - Occupation data with title and description
 */
function displayOccupationInfo(occupation) {
  const infoSection = document.getElementById('occupation-info');
  const titleEl = document.getElementById('occupation-title');
  const descEl = document.getElementById('occupation-description');
  
  if (occupation) {
    titleEl.textContent = occupation.title;
    descEl.textContent = occupation.description || 'No description available.';
    infoSection.style.display = 'block';
  } else {
    infoSection.style.display = 'none';
  }
}

/**
 * Display results in content boxes
 * @param {string} boxId - ID of content box (e.g., 'knowledge-box')
 * @param {Object} data - API response data with results and has_more
 */
function displayResults(boxId, data) {
  const box = document.getElementById(boxId);
  if (!box) return;
  
  // Clear existing content
  box.innerHTML = '';
  
  const items = data.results;
  const hasMore = data.has_more;
  
  if (!items || items.length === 0) {
    // Show empty state
    box.innerHTML = `
      <div class="empty">
        <svg width="36" height="36" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="1.6"></circle>
          <line x1="16.65" y1="16.65" x2="21" y2="21" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"></line>
        </svg>
        <div>No results found. Try adjusting your filters.</div>
      </div>
    `;
    
    // Update count badge
    updateCountBadge(boxId, 0);
    return;
  }
  
  // Determine entity type from boxId
  const entityType = boxId.replace('-box', '');
  
  // Create result items
  items.forEach(item => {
    const itemEl = document.createElement('div');
    itemEl.className = 'result-item';
    itemEl.dataset.id = item.id;
    itemEl.dataset.name = item.name;
    itemEl.dataset.type = entityType;
    
    itemEl.innerHTML = `
      <div class="result-item__header">
        <h3 class="result-item__title">${item.name}</h3>
      </div>
    `;
    
    // Add click handler to open relationship modal
    itemEl.addEventListener('click', () => {
      openRelationshipModal(item.id, item.name, entityType);
    });
    
    box.appendChild(itemEl);
  });
  
  // Add "Load More" button if there are more results
  if (hasMore) {
    const loadMoreBtn = document.createElement('button');
    loadMoreBtn.className = 'load-more-btn';
    loadMoreBtn.textContent = 'Load More';
    loadMoreBtn.onclick = () => loadMoreResults(entityType);
    box.appendChild(loadMoreBtn);
  }
  
  // Update count badge with total count from API
  updateCountBadge(boxId, data.total);
}

/**
 * Update the count badge for a content section
 * @param {string} boxId - ID of content box
 * @param {number} count - Number of items
 */
function updateCountBadge(boxId, count) {
  const box = document.getElementById(boxId);
  if (!box) return;
  
  const section = box.closest('.block');
  const badge = section?.querySelector('.count-badge');
  if (badge) {
    badge.textContent = count;
  }
}

/**
 * Enhanced notification system
 * @param {string} message - Notification message
 * @param {string} type - Notification type ('success', 'error', 'info')
 */
function showNotification(message, type = 'success') {
  const notification = document.createElement('div');
  notification.textContent = message;
  
  const colors = {
    success: '#10b981',
    error: '#ef4444',
    info: '#3b82f6'
  };
  
  Object.assign(notification.style, {
    position: 'fixed',
    top: '24px',
    right: '24px',
    background: colors[type] || colors.success,
    color: '#fff',
    padding: '12px 20px',
    borderRadius: '8px',
    boxShadow: '0 10px 24px rgba(0,0,0,0.15)',
    zIndex: '1000',
    fontSize: '14px',
    fontWeight: '600'
  });
  
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.transition = 'opacity 300ms';
    notification.style.opacity = '0';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}


// ========================================
// AUTOCOMPLETE FUNCTIONALITY
// ========================================

let autocompleteTimeout = null;
let currentSuggestionIndex = -1;

/**
 * Fetch occupation suggestions from API
 */
async function fetchSuggestions(query) {
  if (!query || query.length < 2) {
    hideAutocomplete();
    return;
  }
  
  try {
    // Show loading state
    showAutocompleteLoading();
    
    const response = await fetch(
      `${API_BASE_URL}/api/search/suggest?q=${encodeURIComponent(query)}&limit=10`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    displaySuggestions(data.suggestions, query);
    
  } catch (error) {
    console.error('Failed to fetch suggestions:', error);
    hideAutocomplete();
  }
}

/**
 * Display suggestions in dropdown
 */
function displaySuggestions(suggestions, query) {
  const dropdown = document.getElementById('autocomplete-dropdown');
  const list = document.getElementById('autocomplete-list');
  const loading = document.getElementById('autocomplete-loading');
  const empty = document.getElementById('autocomplete-empty');
  
  // Hide loading
  loading.style.display = 'none';
  
  // Clear previous suggestions
  list.innerHTML = '';
  currentSuggestionIndex = -1;
  
  if (suggestions.length === 0) {
    empty.style.display = 'block';
    list.style.display = 'none';
    dropdown.style.display = 'block';
    return;
  }
  
  empty.style.display = 'none';
  list.style.display = 'block';
  
  // Create suggestion items
  suggestions.forEach((suggestion, index) => {
    const li = document.createElement('li');
    li.dataset.index = index;
    li.dataset.title = suggestion.title;
    
    // Highlight matching text
    const title = highlightMatch(suggestion.title, query);
    li.innerHTML = title;
    
    // Click handler
    li.addEventListener('click', () => {
      selectSuggestion(suggestion.title);
    });
    
    list.appendChild(li);
  });
  
  dropdown.style.display = 'block';
}

/**
 * Highlight matching text in suggestion
 */
function highlightMatch(text, query) {
  const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
  return text.replace(regex, '<strong>$1</strong>');
}

/**
 * Escape regex special characters
 */
function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Show loading state
 */
function showAutocompleteLoading() {
  const dropdown = document.getElementById('autocomplete-dropdown');
  const loading = document.getElementById('autocomplete-loading');
  const list = document.getElementById('autocomplete-list');
  const empty = document.getElementById('autocomplete-empty');
  
  loading.style.display = 'block';
  list.style.display = 'none';
  empty.style.display = 'none';
  dropdown.style.display = 'block';
}

/**
 * Hide autocomplete dropdown
 */
function hideAutocomplete() {
  const dropdown = document.getElementById('autocomplete-dropdown');
  if (dropdown) {
    dropdown.style.display = 'none';
  }
  currentSuggestionIndex = -1;
}

/**
 * Select a suggestion
 */
function selectSuggestion(title) {
  const searchInput = document.getElementById('search');
  searchInput.value = title;
  hideAutocomplete();
  
  // Trigger search
  document.querySelector('.icon-btn').click();
}

/**
 * Handle keyboard navigation
 */
function handleAutocompleteKeyboard(e) {
  const dropdown = document.getElementById('autocomplete-dropdown');
  const list = document.getElementById('autocomplete-list');
  
  if (!dropdown || dropdown.style.display === 'none') {
    return;
  }
  
  const items = list.querySelectorAll('li');
  
  if (items.length === 0) {
    return;
  }
  
  // Arrow Down
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    currentSuggestionIndex = Math.min(currentSuggestionIndex + 1, items.length - 1);
    updateActiveSuggestion(items);
  }
  
  // Arrow Up
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    currentSuggestionIndex = Math.max(currentSuggestionIndex - 1, -1);
    updateActiveSuggestion(items);
  }
  
  // Enter
  if (e.key === 'Enter' && currentSuggestionIndex >= 0) {
    e.preventDefault();
    const selectedItem = items[currentSuggestionIndex];
    selectSuggestion(selectedItem.dataset.title);
  }
  
  // Escape
  if (e.key === 'Escape') {
    hideAutocomplete();
  }
}

/**
 * Update active suggestion visual state
 */
function updateActiveSuggestion(items) {
  items.forEach((item, index) => {
    if (index === currentSuggestionIndex) {
      item.classList.add('active');
      item.scrollIntoView({ block: 'nearest' });
    } else {
      item.classList.remove('active');
    }
  });
}

/**
 * Initialize autocomplete
 */
function initAutocomplete() {
  const searchInput = document.getElementById('search');
  
  if (!searchInput) return;
  
  // Input event with debouncing
  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    
    // Clear previous timeout
    if (autocompleteTimeout) {
      clearTimeout(autocompleteTimeout);
    }
    
    // Debounce: wait 300ms after user stops typing
    autocompleteTimeout = setTimeout(() => {
      fetchSuggestions(query);
    }, 300);
  });
  
  // Keyboard navigation
  searchInput.addEventListener('keydown', handleAutocompleteKeyboard);
  
  // Hide dropdown when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-wrap')) {
      hideAutocomplete();
    }
  });
  
  // Prevent dropdown from closing when clicking inside it
  const dropdown = document.getElementById('autocomplete-dropdown');
  if (dropdown) {
    dropdown.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  }
}


// ========================================
// RELATIONSHIP MODAL FUNCTIONALITY
// ========================================

/**
 * Open relationship modal for an entity
 */
async function openRelationshipModal(entityId, entityName, entityType) {
  const modal = document.getElementById('relationship-modal');
  const titleEl = document.getElementById('modal-entity-name');
  const bodyEl = document.getElementById('modal-body');
  
  // Show modal with loading state
  titleEl.textContent = entityName;
  bodyEl.innerHTML = '<div class="modal-loading">Loading relationships...</div>';
  modal.style.display = 'flex';
  
  try {
    // Fetch relationships from API
    const response = await fetch(
      `${API_BASE_URL}/api/entities/${entityType}/${entityId}/relationships`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Display relationships
    displayRelationships(data);
    
  } catch (error) {
    console.error('Failed to load relationships:', error);
    bodyEl.innerHTML = `
      <div class="relationship-empty">
        Failed to load relationships. Please try again.
      </div>
    `;
  }
}

/**
 * Display relationships in modal
 */
function displayRelationships(data) {
  const bodyEl = document.getElementById('modal-body');
  
  const relationships = data.relationships;
  const hasRelationships = Object.values(relationships).some(rel => rel.count > 0);
  
  if (!hasRelationships) {
    bodyEl.innerHTML = `
      <div class="relationship-empty">
        No relationships found for this ${data.entity.type.slice(0, -1)}.
      </div>
    `;
    return;
  }
  
  // Build HTML for each relationship type
  let html = '';
  
  for (const [relType, relData] of Object.entries(relationships)) {
    if (relData.count === 0) continue;
    
    html += `
      <div class="relationship-section">
        <div class="relationship-header">
          <h3>${relData.header}</h3>
          <span class="relationship-count">${relData.count}</span>
        </div>
        <div class="relationship-items">
          ${relData.items.map(item => `
            <div class="relationship-item" 
                 data-id="${item.id}" 
                 data-name="${item.name}"
                 data-type="${relType}">
              ${item.name}
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  bodyEl.innerHTML = html;
  
  // Add click handlers to relationship items (for nested navigation)
  bodyEl.querySelectorAll('.relationship-item').forEach(item => {
    item.addEventListener('click', () => {
      const itemId = item.dataset.id;
      const itemName = item.dataset.name;
      const itemType = item.dataset.type;
      openRelationshipModal(itemId, itemName, itemType);
    });
  });
}

/**
 * Close relationship modal
 */
function closeRelationshipModal() {
  const modal = document.getElementById('relationship-modal');
  modal.style.display = 'none';
}

/**
 * Initialize modal event listeners
 */
function initRelationshipModal() {
  // Close button
  const closeBtn = document.getElementById('modal-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', closeRelationshipModal);
  }
  
  // Click outside modal to close
  const modal = document.getElementById('relationship-modal');
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeRelationshipModal();
      }
    });
  }
  
  // Escape key to close
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeRelationshipModal();
    }
  });
}

// Load dimensions on page load
loadDimensions();