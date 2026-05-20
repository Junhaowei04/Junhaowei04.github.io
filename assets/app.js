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
            <div class="post-meta">發表於 ${post.date} | ${post.category}</div>
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

function renderTags() {
  if (!(content.tags || []).length) {
    tagCloud.innerHTML = `<span class="empty-line">暂无标签</span>`;
    return;
  }

  tagCloud.innerHTML = (content.tags || [])
    .map((tag, index) => `<a href="#" style="font-size:${0.78 + (index % 5) * 0.08}rem">${tag}</a>`)
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
  const tagCount = (content.tags || []).length;
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
