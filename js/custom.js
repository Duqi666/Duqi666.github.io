document.addEventListener('DOMContentLoaded', function () {

    const element = document.querySelector('#menus a.site-page.child[href="javascript:void(0)"]');
    console.log(element);
    element.addEventListener('click', function (event) {
        console.log('asas')
        // 创建一个 style 元素
        const styleElement = document.createElement('style');
        document.head.appendChild(styleElement);
        const styleSheet = styleElement.sheet;

        // 添加 .card-widget, .wow 样式规则
        styleSheet.insertRule('.card-widget, .wow { background-color: rgba(255, 255, 255, 0.1) !important; border-radius: 8px !important; border: 2px solid rgba(255, 255, 255, 0.5); }', styleSheet.cssRules.length);

        // 添加 .article-title, .content, .author-info-name, .headline, .length-num, .card-widget 样式规则
        styleSheet.insertRule('.article-title, .content, .author-info-name, .headline, .length-num, .card-widget { color: #FFFFFF !important; }', styleSheet.cssRules.length);

        // 添加 .card-archive-list-date, .card-archive-list-count, a.title, .aside-list-item, .toc-link 样式规则
        styleSheet.insertRule('.card-archive-list-date, .card-archive-list-count, a.title, .aside-list-item, .toc-link { color: #FFFFFF !important; }', styleSheet.cssRules.length);

        // 添加 .article-meta-wrap, time 样式规则
        styleSheet.insertRule('.article-meta-wrap, time { color: rgb(205, 205, 205) !important; }', styleSheet.cssRules.length);

        // 添加 .site-name, .menus_item 样式规则
        styleSheet.insertRule('.site-name, .menus_item { font-size: 30px !important; }', styleSheet.cssRules.length);

        // 添加 #site-title 样式规则
        styleSheet.insertRule('#site-title { font-size: 50px !important; }', styleSheet.cssRules.length);

        // 添加 #post 样式规则
        styleSheet.insertRule('#post { background-color: rgba(255, 255, 255, 0.1) !important; color: #FFFFFF !important; border-radius: 8px !important; border: 2px solid rgba(255, 255, 255, 0.5); }', styleSheet.cssRules.length);

        // 添加 h1, h2, h3 样式规则
        styleSheet.insertRule('h1, h2, h3 { color: #FFFFFF !important; }', styleSheet.cssRules.length);



    });

});