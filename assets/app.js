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

function renderPosts() {
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

  postList.innerHTML = posts
    .map(
      (post, index) => `
        <article class="post-card ${index % 2 ? "is-reverse" : ""}">
            <div class="post-cover ${post.cover}" aria-hidden="true"></div>
          <div class="post-content">
            <a class="post-title" href="#">${post.title}</a>
            <div class="post-meta">发布于 ${post.date} | ${post.category}</div>
            <p>${post.summary}</p>
            <a class="read-more" href="#">阅读全文</a>
          </div>
        </article>
      `,
    )
    .join("");
  pagination.hidden = posts.length <= 5;
}

function renderLatest() {
  if (!posts.length) {
    latestList.innerHTML = `<li class="empty-line">暂无文章</li>`;
    return;
  }

  latestList.innerHTML = posts
    .slice(0, 5)
    .map((post) => `<li><a href="#">${post.title}</a><time>${post.date}</time></li>`)
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
  const categories = content.categories || [];

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

  archiveList.innerHTML = posts
    .map((post) => `<li><a href="#">${post.date} <span>1</span></a></li>`)
    .join("");
}

function renderStats() {
  const categories = content.categories || [];
  const tagCount = getTagStats().length;
  const lastPost = posts[0];

  document.querySelector("#author-post-count").textContent = posts.length;
  document.querySelector("#author-tag-count").textContent = tagCount;
  document.querySelector("#author-category-count").textContent = categories.length;
  document.querySelector("#site-post-count").textContent = posts.length;
  document.querySelector("#last-updated").textContent = lastPost?.date || "--";
}

function syncScrollState() {
  const active = window.scrollY > 60;
  topNav.classList.toggle("is-scrolled", active);
  backTop.classList.toggle("is-visible", window.scrollY > window.innerHeight * 0.7);
}

renderPosts();
renderLatest();
renderTags();
renderCategories();
renderArchive();
renderStats();
syncScrollState();

window.addEventListener("scroll", syncScrollState, { passive: true });
backTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
