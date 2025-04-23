const lowResBgUrl_dark = '/imgs/output_low.png';
const highResBgUrl_dark = '/imgs/output.png';
const highResBgUrl_light_white = '/imgs/cloud_white.jpg'
const highResBgUrl_light = '/imgs/cloud.jpg'
const link = document.createElement('link');
link.rel = 'stylesheet';
link.href = '/css/mode_change.css';
document.head.appendChild(link);

function mode_change_preloader(interval_time){
    const $loadingBox = document.getElementById('loading-box')
    const $body = document.body
    const preloader = {
      endLoading: () => {
        $body.style.overflow = ''
        $loadingBox.classList.add('loaded')
      },
      initLoading: () => {
        $body.style.overflow = 'hidden'
        $loadingBox.classList.remove('loaded')
      }
    }

    preloader.initLoading()
    setTimeout(preloader.endLoading,interval_time)
}

// 查找触发元素并添加点击事件监听器
function setupClickEvent() {
    const triggerElementDark = document.querySelector('#menus a.site-page.child[href="javascript:void(0)"]');
    if (triggerElementDark) {
        triggerElementDark.addEventListener('click', function () {
            document.documentElement.setAttribute('mode', 'dark');
            localStorage.setItem('Mode', 'dark');
            document.body.classList.add('mode_change');
            load_background_img(lowResBgUrl_dark,lowResBgUrl_dark,highResBgUrl_dark,highResBgUrl_dark)
            mode_change_preloader(300)
        });
    }
    const triggerElementLight = document.querySelector('#menus a.site-page.child[href="javascript:void(1)"]');
    if (triggerElementLight) {
        triggerElementLight.addEventListener('click', function () {
            localStorage.setItem('Mode', 'light');
            document.documentElement.setAttribute('mode', 'light');
            document.body.classList.remove('mode_change');
            load_background_img(highResBgUrl_light,highResBgUrl_light_white,highResBgUrl_light,highResBgUrl_light_white)
            mode_change_preloader(300)
        });

    }
}



// 页面加载时检查 LocalStorage 状态并应用样式
// function applyStyleBasedOnStorage() {

//     const isStyleApplied = localStorage.getItem('customStyleApplied') === 'true';
//     if (isStyleApplied) {
//         document.body.classList.add('mode_change');
//     }
// }

// 页面加载完成后执行操作
document.addEventListener('DOMContentLoaded', function () {
    // addStyleSheet(styleRules);
    // applyStyleBasedOnStorage();
    setupClickEvent();
    // style_apply();

});


window.addEventListener('DOMContentLoaded', function () {
    const currentPage = window.location.pathname;
    const regex = /^\/(19|20)\d{2}\/(0[1-9]|1[0-2])\/(0[1-9]|[12][0-9]|3[01])\/[^/]+\/$/;
    let is_paper = regex.test(currentPage);
    let Mode = localStorage.getItem('Mode')

    if(is_paper && Mode === 'dark'){
        document.body.classList.remove('mode_change');
    }
    else if(is_paper && Mode != 'dark'){
        document.body.classList.remove('mode_change');
    }
    else if( Mode === 'dark'){
        document.body.classList.add('mode_change');
    }else{
        document.body.classList.remove('mode_change');
    }
    document.documentElement.setAttribute('mode', localStorage.getItem('Mode'));
});



// 先设置低分辨率背景图片
function load_background_img(lowUrl,lowUrlWhite,HighUrl,HighUrlWhite){
    const pageHeader = document.getElementById('page-header');
    if (pageHeader) {
        pageHeader.style.backgroundImage = `url('${lowUrl}')`;
        const img = new Image();
        img.src = lowUrl;
        img.onload = function () {
            pageHeader.style.backgroundImage = `url('${HighUrl}')`;
            console.log('gao');
        };
    }
    document.body.style.backgroundImage = `url('${lowUrlWhite}')`;
    // changeBackground(lowUrlWhite)
    const img = new Image();
    // 创建 Image 对象加载高分辨率图片
    img.src = lowUrlWhite;
    // 当高分辨率图片加载完成后，替换背景图片
    img.onload = function () {
        document.body.style.backgroundImage = `url('${HighUrlWhite}')`;
        console.log('gao');
    };

    // 处理图片加载失败的情况

}
// function load_background_img_light(lowResBgUrl,highResBgUrl){
//     const pageHeader = document.getElementById('page-header');
//     if (pageHeader) {
//         pageHeader.style.removeProperty('background');
//         pageHeader.style.backgroundImage = `url('${lowResBgUrl}')`;

//     }
//     document.body.style.backgroundImage = `url('${highResBgUrl}')`;



// }




// function load_background_img(ImgUrl,ImgUrlWhite){
//     document.body.style.backgroundImage = `url('${lowResBgUrl}')`;
// }

if (!localStorage.getItem('isFirstVisit')) {
    // 首次访问，设置 Mode 为 dark
    localStorage.setItem('Mode', 'dark');
    document.documentElement.setAttribute('mode', 'dark');
    // 设置 isFirstVisit 标志
    localStorage.setItem('isFirstVisit', 'true');
    load_background_img(lowResBgUrl_dark,lowResBgUrl_dark,highResBgUrl_dark,highResBgUrl_dark)
}else{
    if(localStorage.getItem('Mode')=='dark'){
        load_background_img(lowResBgUrl_dark,lowResBgUrl_dark,highResBgUrl_dark,highResBgUrl_dark)
    }else{
        load_background_img(highResBgUrl_light,highResBgUrl_light_white,highResBgUrl_light,highResBgUrl_light_white)
    }
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
// document.addEventListener('DOMContentLoaded', function () {
//     const currentPage = window.location.pathname;
//     if(currentPage=='/'){
//         const postCoverDivRight = document.querySelectorAll('div.post_cover');
//         for(let i=0;i<postCoverDivRight.length;i++){
//             let post_bg = postCoverDivRight[i].getElementsByClassName('post-bg')
//             postCoverDivRight[i].style.height = post_bg[0].offsetHeight + 'px';
//         }
//     }
// });


// const mainElement = document.querySelector('main');
// const clonedMain = mainElement.cloneNode(false);
// clonedMain.classList.add('cloned-main');
// mainElement.parentNode.insertBefore(clonedMain, mainElement);

// const resizeObserver = new ResizeObserver((entries) => {
//     for (const entry of entries) {
//         if (entry.target === mainElement) {
//             const newHeight = entry.contentRect.height;
//             clonedMain.style.height = newHeight + 'px';
//         }
//     }
// });

// resizeObserver.observe(mainElement);

// let previousTop = mainElement.getBoundingClientRect().top;

// function checkPosition() {
//     const currentTop = mainElement.getBoundingClientRect().top;
//     if (currentTop!== previousTop) {
//         clonedMain.style.top = currentTop + window.scrollY + 'px';
//         previousTop = currentTop;
//     }
//     requestAnimationFrame(checkPosition);
// }

// checkPosition();


const allNodes = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
);
while (allNodes.nextNode()) {
    const node = allNodes.currentNode;
    if (node.textContent.trim() === '-') {
        console.log(node);
        node.textContent = ''
        break;
    }
}

