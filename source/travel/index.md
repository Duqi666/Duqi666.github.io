---
title: travel
date: 2025-04-27 12:23:12
---


<!-- 引入 Leaflet 的 CSS -->
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />

<!-- 引入 Leaflet 的 JavaScript 文件 -->
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>

<style>
    #map {
        height: 700px;
        width: 100%;
        position: relative;
    }
</style>

<div id="map"></div>




<script>
    document.addEventListener('DOMContentLoaded', function() {
        var map = L.map('map').setView([35,100], 4.2);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
        }).addTo(map);

        // fetch('../cities.json')
        //     .then(response => response.json())
        //     .then(data => {
        //         L.geoJSON(data, {
        //             pointToLayer: function (feature, latlng) {
        //                 var color = getColor(feature.properties.visits);
        //                 return L.circleMarker(latlng, { fillColor: color, fillOpacity: 0.5 });
        //             },
        //             onEachFeature: function (feature, layer) {
        //                 layer.bindPopup(`<h2>${feature.properties.name}</h2><img src="${feature.properties.image}" width="200">`);
        //             }
        //         }).addTo(map);
        //     });
        let markers = [];
        fetch('../cities.json')
            .then(response => response.json())
            .then(data => {
                L.geoJSON(data, {
                    pointToLayer: function (feature, latlng) {
                        var color = getColor(feature.properties.visits);
                        var iconSize = getIconSize(map.getZoom());
                        var emoji = feature.properties.emoji;
                        var icon = L.divIcon({
                            className: 'custom-icon',
                            html: `<span style="color: ${color}; font-size: ${iconSize}px;">${emoji}</span>`,
                        });
                        var marker = L.marker(latlng, { icon: icon });
                        markers.push(marker);
                        return marker;
                    },
                    onEachFeature: function (feature, layer) {
                        layer.bindPopup(`<h2>${feature.properties.name}</h2><img src="${feature.properties.image}" width="200">`);
                    }
                }).addTo(map);
            });

        function getColor(visits) {
            // 根据你访问的次数返回不同的颜色
            return visits > 10 ? '#800026' :
                   visits > 5  ? '#BD0026' :
                   visits > 2  ? '#E31A1C' :
                   visits > 1  ? '#FC4E2A' :
                                 '#FFEDA0';
        }
        function getIconSize(zoom) {
            // 简单示例：缩放级别每增加 1，图标大小增加 2 像素
            var size = 10 + (zoom - 4) * 2
            if(size<2){
                return 2
            }else{
                return size
            }
        }

        map.on('zoomend', function () {
            var zoom = map.getZoom();
            markers.forEach(function (marker) {
                var color = getColor(marker.feature.properties.visits);
                var iconSize = getIconSize(zoom);
                var emoji = marker.feature.properties.emoji;
                console.log(emoji);
                var icon = L.divIcon({
                    className: 'custom-icon',
                    html: `<span style="color: ${color}; font-size: ${iconSize}px;">${emoji}</span>`,
                    iconSize: [iconSize, iconSize]
                });
                marker.setIcon(icon);
            });
        });

        // 强制Leaflet重新计算地图的尺寸
        setTimeout(function() {
            map.invalidateSize();
        }, 100);
    });
</script>