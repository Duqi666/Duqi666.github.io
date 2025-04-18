// 定义样式规则
const styleRules = `
.custom-style .card-widget,
.custom-style .recent-post-item,
.custom-style #archive {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 3px solid #FFF; /* 设置边框的宽度、样式和颜色 /
    border-radius: 5px; / 设置四个角的圆角半径，数值越大圆角越明显 */
}

.custom-style .article-title,
.custom-style .content,
.custom-style .author-info-name,
.custom-style .headline,
.custom-style .length-num,
.custom-style .card-widget,
.custom-style .blog-slider__title,
.custom-style .blog-slider__text {
    color: #FFFFFF !important;
}
.custom-style .card-archive-list-date,
.custom-style .card-archive-list-count,
.custom-style a.title,
.custom-style .aside-list-item,
.custom-style .toc-link,
.custom-style #archive,
.custom-style .article-sort-item-title {
    color: #FFFFFF !important;
}
.custom-style .article-meta-wrap,
.custom-style time,
.custom-style .blog-slider__code{
    color: rgb(205, 205, 205) !important;
}
.custom-style .site-name,
.custom-style .menus_item {
    font-size: 30px !important;
}
.custom-style #site-title {
    font-size: 50px !important;
}

`;

// 创建并添加样式表到页面
function addStyleSheet(rules) {
    const styleElement = document.createElement('style');
    styleElement.type = 'text/css';
    styleElement.textContent = rules;
    document.head.appendChild(styleElement);
}

// 查找触发元素并添加点击事件监听器
function setupClickEvent() {
    const triggerElementDark = document.querySelector('#menus a.site-page.child[href="javascript:void(0)"]');
    if (triggerElementDark) {
        triggerElementDark.addEventListener('click', function () {
            localStorage.setItem('customStyleApplied', 'true');
            localStorage.setItem('Mode', 'dark');
            document.body.classList.add('custom-style');
        });
    }
    const triggerElementLight = document.querySelector('#menus a.site-page.child[href="javascript:void(1)"]');
    if (triggerElementLight) {
        triggerElementLight.addEventListener('click', function () {
            localStorage.setItem('customStyleApplied', 'false');
            localStorage.setItem('Mode', 'light');
            document.body.classList.remove('custom-style');
        });
    }
}

function style_apply() {
    const isStyleApplied = localStorage.getItem('customStyleApplied') === 'true';
    if (isStyleApplied) {
        document.body.classList.add('custom-style');
    } else {
        document.body.classList.remove('custom-style');
    }
}



// 页面加载时检查 LocalStorage 状态并应用样式
// function applyStyleBasedOnStorage() {

//     const isStyleApplied = localStorage.getItem('customStyleApplied') === 'true';
//     if (isStyleApplied) {
//         document.body.classList.add('custom-style');
//     }
// }

// 页面加载完成后执行操作
document.addEventListener('DOMContentLoaded', function () {
    addStyleSheet(styleRules);
    // applyStyleBasedOnStorage();
    setupClickEvent();
    // style_apply();

});


window.addEventListener('DOMContentLoaded', function () {
    const currentPage = window.location.pathname;
    const regex = /^\/(19|20)\d{2}\/(0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/[^/]+\/$/;
    let is_paper = regex.test(currentPage);
    console.log(currentPage, is_paper);

    if (is_paper) {
        document.body.classList.remove('custom-style');
    } else {
        if (localStorage.getItem('Mode') === 'dark') {
            document.body.classList.add('custom-style');
        } else {
            document.body.classList.remove('custom-style');
        }
    }
    // console.log(currentPage);
    // let papers = document.getElementsByClassName('article-title')
    // let flag = 0;
    // for(let i = 1;i<papers.length;i++){
    //     if (papers[i].hasAttribute('href') && papers[i].getAttribute('herf')==currentPage){
    //         flag = 1;
    //         console.log('12312312');
    //         break
    //     }
    // }
    // if(flag==1){
    //     localStorage.setItem('customStyleApplied', 'false');
    // }else{
    //     localStorage.setItem('customStyleApplied', 'true');
    // }


});

const lowResBgUrl = '/imgs/I.png';
const highResBgUrl = '/imgs/output.png';

// 先设置低分辨率背景图片
document.body.style.backgroundImage = `url('${lowResBgUrl}')`;

// 创建 Image 对象加载高分辨率图片
const img = new Image();
img.src = highResBgUrl;

// 当高分辨率图片加载完成后，替换背景图片
img.onload = function () {
    document.body.style.backgroundImage = `url('${highResBgUrl}')`;
    console.log('gao');
};

// 处理图片加载失败的情况
img.onerror = function () {
    console.error('Failed to load high resolution background image.');
};

if (!localStorage.getItem('isFirstVisit')) {
    // 首次访问，设置 Mode 为 dark
    localStorage.setItem('Mode', 'dark');
    // 设置 isFirstVisit 标志
    localStorage.setItem('isFirstVisit', 'true');
}


// document.addEventListener('DOMContentLoaded', function () {
//     const currentPage = window.location.pathname;
//     if(currentPage=='/'){
//         const postCoverDivLeft = document.querySelectorAll('div.post_cover.left');
//         for(let i=0;i<postCoverDivLeft.length;i++){
//             let post_bg = postCoverDivLeft[i].getElementsByClassName('post-bg')
//             postCoverDivLeft[i].style.width = post_bg[0].offsetWidth + 'px';
//             post_bg[0].getBoundingClientRect().clientWidth
//             console.log( post_bg[0].clientWidth);
//         }
//         const postCoverDivRight = document.querySelectorAll('div.post_cover.right');
//         for(let i=0;i<postCoverDivRight.length;i++){
//             let post_bg = postCoverDivRight[i].getElementsByClassName('post-bg')
//             postCoverDivRight[i].style.width = post_bg[0].offsetWidth + 'px';

//         }
//     }
// });

// ----index_layout=6时
document.addEventListener('DOMContentLoaded', function () {
    const currentPage = window.location.pathname;
    if(currentPage=='/'){
        const postCoverDivRight = document.querySelectorAll('div.post_cover');
        for(let i=0;i<postCoverDivRight.length;i++){
            let post_bg = postCoverDivRight[i].getElementsByClassName('post-bg')
            postCoverDivRight[i].style.height = post_bg[0].offsetHeight + 'px';
        }
    }
});



