const content = window.siteContent || { posts: [], tags: [] };
const posts = content.posts || [];
const postList = document.querySelector("#post-list");
const latestList = document.querySelector("#latest-list");
const tagCloud = document.querySelector("#tag-cloud");
const categoryList = document.querySelector("#category-list");
const archiveList = document.querySelector("#archive-list");
const pagination = document.querySelector("#pagination");
const topNav = document.querySelector("#top-nav");
const backTop = document.querySelector("#back-top");
const POSTS_PER_PAGE = 2;

function clampPage(page) {
  const totalPages = Math.max(1, Math.ceil(posts.length / POSTS_PER_PAGE));
  return Math.min(Math.max(Number(page) || 1, 1), totalPages);
}

function getPageFromHash() {
  const match = window.location.hash.match(/^#page-(\d+)$/);
  return clampPage(match ? match[1] : 1);
}

function getPostFromHash() {
  const match = window.location.hash.match(/^#post\/(.+)$/);
  if (!match) return null;
  return posts.find((post) => post.slug === decodeURIComponent(match[1])) || null;
}

function renderPosts() {
  const activePost = getPostFromHash();
  if (activePost) {
    renderArticle(activePost);
    return;
  }

  document.title = "MomenT - 个人博客";

  if (!posts.length) {
    postList.innerHTML = `
      <article class="empty-post-card">
        <div class="empty-icon">M</div>
        <h2>暂无文章</h2>
        <p>这里会显示 MomenT 发布的博客、学习笔记和研究记录。</p>
      </article>
    `;
    pagination.hidden = true;
    return;
  }

  const page = getPageFromHash();
  const start = (page - 1) * POSTS_PER_PAGE;
  const pagePosts = posts.slice(start, start + POSTS_PER_PAGE);

  postList.innerHTML = pagePosts
    .map(
      (post, index) => `
        <article class="post-card ${index % 2 ? "is-reverse" : ""}">
          <div class="post-cover ${post.cover}" aria-hidden="true"></div>
          <div class="post-content">
            <a class="post-title" href="#post/${encodeURIComponent(post.slug)}">${post.title}</a>
            <div class="post-meta">发布于 ${post.date} | ${post.category}</div>
            <p>${post.summary}</p>
            <a class="read-more" href="#post/${encodeURIComponent(post.slug)}">阅读全文</a>
          </div>
        </article>
      `,
    )
    .join("");

  renderPagination(page);
}

function renderPagination(activePage) {
  const totalPages = Math.ceil(posts.length / POSTS_PER_PAGE);
  if (totalPages <= 1) {
    pagination.hidden = true;
    pagination.innerHTML = "";
    return;
  }

  pagination.hidden = false;
  pagination.innerHTML = Array.from({ length: totalPages }, (_, index) => {
    const page = index + 1;
    const className = page === activePage ? "is-current" : "";
    return `<a class="${className}" href="#page-${page}" aria-label="第 ${page} 页">${page}</a>`;
  }).join("");
}

function renderArticle(post) {
  document.title = `${post.title} - MomenT`;
  pagination.hidden = true;
  pagination.innerHTML = "";
  postList.innerHTML = `
    <article class="article-card">
      <a class="back-link" href="#content">← 返回文章列表</a>
      <header class="article-header">
        <p class="post-meta">发布于 ${post.date} | ${post.category}</p>
        <h1>${post.title}</h1>
        <p>${post.summary}</p>
        <div class="article-tags">${(post.tags || []).map((tag) => `<span>${tag}</span>`).join("")}</div>
      </header>
      <div class="article-body">${post.body || "<p>文章正在整理中。</p>"}</div>
    </article>
  `;
  typesetMath(postList);
  postList.scrollIntoView({ behavior: "smooth", block: "start" });
}

function typesetMath(root, attempt = 0) {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetClear?.([root]);
    window.MathJax.typesetPromise([root]).catch(() => {});
    return;
  }

  if (attempt < 40) {
    window.setTimeout(() => typesetMath(root, attempt + 1), 120);
  }
}

function renderLatest() {
  if (!posts.length) {
    latestList.innerHTML = `<li class="empty-line">暂无文章</li>`;
    return;
  }

  latestList.innerHTML = posts
    .slice(0, 5)
    .map((post) => `<li><a href="#post/${encodeURIComponent(post.slug)}">${post.title}</a><time>${post.date}</time></li>`)
    .join("");
}

function getTagStats() {
  const counts = new Map();

  posts.forEach((post) => {
    (post.tags || []).forEach((tag) => {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    });
  });

  (content.tags || []).forEach((tag) => {
    if (!counts.has(tag)) counts.set(tag, 0);
  });

  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, "zh-CN"));
}

function renderTags() {
  const tags = getTagStats();

  if (!tags.length) {
    tagCloud.innerHTML = `<span class="empty-line">暂无关键词</span>`;
    return;
  }

  const maxCount = Math.max(...tags.map((tag) => tag.count), 1);
  const minCount = Math.min(...tags.map((tag) => tag.count));
  const colors = ["#2f80ed", "#ff6b3d", "#16b4a5", "#6c5ce7", "#30a46c", "#b454a0"];

  tagCloud.innerHTML = tags
    .map((tag, index) => {
      const ratio = tag.count / maxCount;
      const balanced = maxCount === minCount ? 0.48 + (index % 4) * 0.09 : ratio;
      const size = 0.95 + balanced * 1.25;
      const weight = Math.round(520 + balanced * 260);
      const shift = [-4, 8, -1, 5, -7, 3][index % 6];
      const color = colors[index % colors.length];
      const countText = tag.count ? `${tag.count} 篇相关文章` : "暂无相关文章";

      return `<a href="#" title="${countText}" style="--size:${size.toFixed(2)}rem; --weight:${weight}; --shift:${shift}px; --color:${color}">${tag.name}</a>`;
    })
    .join("");
}

function renderCategories() {
  const counts = new Map();
  posts.forEach((post) => counts.set(post.category, (counts.get(post.category) || 0) + 1));
  const categories = [...counts.entries()].map(([name, count]) => ({ name, count }));

  if (!categories.length) {
    categoryList.innerHTML = `<li class="empty-line">暂无分类</li>`;
    return;
  }

  categoryList.innerHTML = categories
    .map((category) => `<li><a href="#">${category.name} <span>${category.count}</span></a></li>`)
    .join("");
}

function renderArchive() {
  if (!posts.length) {
    archiveList.innerHTML = `<li class="empty-line">暂无归档</li>`;
    return;
  }

  const archive = new Map();
  posts.forEach((post) => {
    const key = post.date.slice(0, 7);
    archive.set(key, (archive.get(key) || 0) + 1);
  });

  archiveList.innerHTML = [...archive.entries()]
    .map(([date, count]) => `<li><a href="#">${date} <span>${count}</span></a></li>`)
    .join("");
}

function renderStats() {
  const categoryCount = new Set(posts.map((post) => post.category)).size;
  const tagCount = getTagStats().length;
  const lastPost = posts[0];

  document.querySelector("#author-post-count").textContent = posts.length;
  document.querySelector("#author-tag-count").textContent = tagCount;
  document.querySelector("#author-category-count").textContent = categoryCount;
  document.querySelector("#site-post-count").textContent = posts.length;
  document.querySelector("#last-updated").textContent = lastPost?.date || "--";
}

function syncScrollState() {
  const active = window.scrollY > 60;
  topNav.classList.toggle("is-scrolled", active);
  backTop.classList.toggle("is-visible", window.scrollY > window.innerHeight * 0.7);
}

function renderApp() {
  renderPosts();
  renderLatest();
  renderTags();
  renderCategories();
  renderArchive();
  renderStats();
  syncScrollState();
}

renderApp();

window.addEventListener("hashchange", renderApp);
window.addEventListener("scroll", syncScrollState, { passive: true });
backTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
