// // document.addEventListener('DOMContentLoaded', function () {


// //     let article_title = this.getElementsByClassName('article-title')
// //     for(i=0;i<article_title.length;i++){
// //         article_title[i].addEventListener('click',function(e){
// //             const styleElement = document.createElement('style');
// //             document.head.appendChild(styleElement);
// //             const styleSheet = styleElement.sheet;
    
// //             // 添加 .card-widget, .wow 样式规则
// //             styleSheet.insertRule('.card-widget, .wow { background-color: rgba(255, 255, 255, 1) !important; border-radius: 8px !important; border: 2px solid rgba(255, 255, 255, 0.5); }', styleSheet.cssRules.length);
    
// //         })
// //     }


// // });

// document.addEventListener('DOMContentLoaded', function () {
//     // 获取按钮元素
//     let article_title = this.getElementsByClassName('article-title')

//     if (article_title) {
//         // 为按钮添加点击事件监听器
//         article_title.addEventListener('click', function () {
//             // 存储状态到 localStorage
//             localStorage.setItem('shouldChangeH1Color', 'true');
//             // 这里可以替换为你实际要跳转的页面 URL
//         });
//     }

//     // 检查 localStorage 中的状态
//     const shouldChangeColor = localStorage.getItem('shouldChangeH1Color');
//     if (shouldChangeColor === 'true') {
//         // 获取所有 h1 元素
//         const styleElement = document.createElement('style');
//         document.head.appendChild(styleElement);
//         const styleSheet = styleElement.sheet;

//         // 添加 .card-widget, .wow 样式规则
//         styleSheet.insertRule('.card-widget, .wow { background-color: rgba(255, 255, 255, 1) !important; border-radius: 8px !important; border: 2px solid rgba(255, 255, 255, 0.5); }', styleSheet.cssRules.length);

//         // 清除 localStorage 中的状态标记
//         localStorage.removeItem('shouldChangeH1Color');
//     }
// });    
const articleTitles = document.getElementsByClassName('article-title');
for (let i = 0; i < articleTitles.length; i++) {
    articleTitles[i].addEventListener('click', function () {
        // 存储标志到 localStorage
        localStorage.setItem('changeH1Color', 'true');
    });
}

// 在新页面加载时检查 localStorage 并修改 h1 颜色
// window.addEventListener('load', function () {
//     localStorage.setItem('customStyleApplied', 'false');
//     // const shouldChangeColor = localStorage.getItem('changeH1Color');
//     // if (shouldChangeColor === 'true') {
//     //     const h1Elements = document.getElementsByTagName('h1');
//     //     for (let i = 0; i < h1Elements.length; i++) {
//     //         h1Elements[i].style.color = 'purple';
//     //     }
//     //     // 移除标志
//     //         const styleElement = document.createElement('style');
//     //         document.head.appendChild(styleElement);
//     //         const styleSheet = styleElement.sheet;

//     //         // 添加 .card-widget, .wow 样式规则
//     //         styleSheet.insertRule('.card-widget, .wow { background-color: rgba(255, 255, 255, 1) !important; border-radius: 8px !important; border: 2px solid rgba(255, 255, 255, 0.5); }', styleSheet.cssRules.length);

//     //     localStorage.removeItem('changeH1Color');
//     }
// });    

