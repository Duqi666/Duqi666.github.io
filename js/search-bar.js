(function() {
  'use strict';

  /* ===== 工具函数 ===== */
  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function stripHtml(html) {
    var tmp = document.createElement('div');
    tmp.innerHTML = html;
    return (tmp.textContent || tmp.innerText || '').replace(/\s+/g, ' ').trim();
  }

  function highlightKeyword(text, keyword) {
    var escaped = escapeHtml(text);
    if (!keyword) return escaped;
    var escapedKeyword = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    var regex = new RegExp('(' + escapedKeyword + ')', 'gi');
    return escaped.replace(regex, '<mark class="search-highlight">$1</mark>');
  }

  /* ===== 搜索栏插入 ===== */
  function insertSearchBar() {
    if (document.querySelector('.custom-search-bar')) return;

    var contentInner = document.getElementById('content-inner');
    if (!contentInner) return;

    var isSearchPage = window.location.pathname.indexOf('/search/') !== -1;

    var currentQuery = '';
    if (isSearchPage) {
      var params = new URLSearchParams(window.location.search);
      currentQuery = params.get('q') || '';
    }

    var searchBar = document.createElement('div');
    searchBar.className = 'custom-search-bar';
    searchBar.innerHTML =
      '<div class="search-bar-container">' +
        '<div class="search-bar-inner">' +
          '<i class="fas fa-search search-bar-icon"></i>' +
          '<input type="text" class="search-bar-input" placeholder="搜索文章..." value="' + currentQuery.replace(/"/g, '&quot;') + '" />' +
          '<button class="search-bar-btn">搜索</button>' +
        '</div>' +
      '</div>';

    contentInner.parentNode.insertBefore(searchBar, contentInner);

    var input = searchBar.querySelector('.search-bar-input');
    var btn = searchBar.querySelector('.search-bar-btn');

    function doSearch() {
      var q = input.value.trim();
      if (q) {
        window.location.href = '/search/?q=' + encodeURIComponent(q);
      }
    }

    btn.addEventListener('click', doSearch);
    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') doSearch();
    });
  }

  /* ===== 搜索结果页面 ===== */
  function initSearchPage() {
    var params = new URLSearchParams(window.location.search);
    var query = params.get('q') || '';

    var contentInner = document.getElementById('content-inner');
    if (!contentInner) return;

    contentInner.style.display = 'none';

    var resultsContainer = document.createElement('div');
    resultsContainer.className = 'search-results-container';

    if (!query) {
      resultsContainer.innerHTML =
        '<div class="search-results-header">' +
          '<div class="search-results-title">搜索文章</div>' +
          '<div class="search-results-info">在上方输入关键词开始搜索</div>' +
        '</div>';
      contentInner.parentNode.insertBefore(resultsContainer, contentInner);
      return;
    }

    resultsContainer.innerHTML =
      '<div class="search-results-header">' +
        '<div class="search-results-title">搜索结果</div>' +
        '<div class="search-results-info">搜索中...</div>' +
      '</div>' +
      '<div class="search-loading">正在加载搜索数据</div>';
    contentInner.parentNode.insertBefore(resultsContainer, contentInner);

    fetch('/search.xml')
      .then(function(res) { return res.text(); })
      .then(function(text) {
        var parser = new DOMParser();
        var xmlDoc = parser.parseFromString(text, 'text/xml');
        var entries = xmlDoc.querySelectorAll('entry');

        var results = [];
        var lowerQuery = query.toLowerCase();

        entries.forEach(function(entry) {
          var title = entry.querySelector('title') ? entry.querySelector('title').textContent : '';
          var url = entry.querySelector('url') ? entry.querySelector('url').textContent : '';
          var rawContent = entry.querySelector('content') ? entry.querySelector('content').textContent : '';
          var content = stripHtml(rawContent);

          var lowerTitle = title.toLowerCase();
          var lowerContent = content.toLowerCase();

          var titleMatch = lowerTitle.indexOf(lowerQuery) !== -1;
          var contentMatch = lowerContent.indexOf(lowerQuery) !== -1;

          if (titleMatch || contentMatch) {
            var excerpt = '';
            if (contentMatch) {
              var idx = lowerContent.indexOf(lowerQuery);
              var start = Math.max(0, idx - 60);
              var end = Math.min(content.length, idx + query.length + 120);
              excerpt = (start > 0 ? '...' : '') + content.substring(start, end) + (end < content.length ? '...' : '');
            } else {
              excerpt = content.substring(0, 150) + (content.length > 150 ? '...' : '');
            }

            results.push({
              title: title,
              url: url,
              excerpt: excerpt
            });
          }
        });

        renderSearchResults(results, query);
      })
      .catch(function() {
        resultsContainer.innerHTML =
          '<div class="search-results-header">' +
            '<div class="search-results-title">搜索结果</div>' +
          '</div>' +
          '<div class="search-error">搜索数据加载失败，请稍后重试</div>';
      });
  }

  function renderSearchResults(results, query) {
    var container = document.querySelector('.search-results-container');
    if (!container) return;

    var header =
      '<div class="search-results-header">' +
        '<div class="search-results-title">搜索结果</div>' +
        '<div class="search-results-info">关于 "<span class="search-query">' + escapeHtml(query) + '</span>" 找到 <span class="search-count">' + results.length + '</span> 篇文章</div>' +
      '</div>';

    var list = '';
    if (results.length > 0) {
      list = '<div class="search-results-list">';
      results.forEach(function(r) {
        list +=
          '<a href="' + r.url + '" class="search-result-card">' +
            '<div class="search-result-title">' + highlightKeyword(r.title, query) + '</div>' +
            '<div class="search-result-excerpt">' + highlightKeyword(r.excerpt, query) + '</div>' +
          '</a>';
      });
      list += '</div>';
    } else {
      list =
        '<div class="search-no-results">' +
          '<div class="search-no-results-icon"><i class="fas fa-search"></i></div>' +
          '<p>未找到与 "<span class="search-query">' + escapeHtml(query) + '</span>" 相关的文章</p>' +
          '<p class="search-no-results-hint">试试其他关键词？</p>' +
        '</div>';
    }

    container.innerHTML = header + list;
  }

  /* ===== 初始化 ===== */
  function init() {
    var isSearchPage = window.location.pathname.indexOf('/search/') !== -1;
    if (isSearchPage) {
      insertSearchBar();
      initSearchPage();
    } else {
      insertSearchBar();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  document.addEventListener('pjax:complete', init);
  document.addEventListener('pjax:success', init);
})();
