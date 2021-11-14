import glob 
data_file_list= glob.glob('same_geo/*.DATA')

include_files = [873,901,912,1,1280,1430,2467,2466,3900,3901,3866,3868,1027,3839,1042,1058,1073,1242,1291,1770,1089,1103,1135,1150,1166,1181,1197,1211,1227,3852,3832,3848,3864,3833,3879,3849,3853,3834,3867,3835,385,3865,3854,3836,3869,3850,3837,3863,1590,1740,190,1011,1506,2541,2020,2040,2450,2600,2760,2910,3060,1606,1605,3210,3370,3520,3680,4040,420,4350,4500,4660,2172,4810,4970,5110,5550,5700,5860,274,3522,3534,632,648,663,840,715,730,746,761,777,792,3240,807,906,874,856,887,896,857,875,888,858,902,876,859,4117,897,889,86,877,436,4414,4411,4413,860,907,878,4662,4709,4706,4708,861,89,898,903,862,879,4972,5000,500,5003,91,911,5272,863,890,88,864,899,865,891,904,880,908,892,866,881,9,867,882,868,893,910,905,869,883,90,913,87,894,884,871,909,872,885,617,895,230,915,914,918,917,92,919,921,920,923,922,925,924,927,926,929,928,930,93,933,932,935,934,937,936,939,938,940,94,942,941,944,943,946,945,949,948,950,95,952,951,954,953,956,955,958,957,96,959,961,960,964,963,966,965,968,967,97,969,971,970,973,972,975,974,977,976,98,979,981,980,983,982,985,984,987,986,989,988,990,99,992,991,994,993,996,995,998,997,1012,999,1014,1013,1016,1015,1018,1017,102,1019,1021,1020,1023,1022,1028,1024,103,1029,1031,1030,1033,1032,1035,1034,1037,1036,1039,1038,1043,104,1045,1044,1047,1046,1049,1048,1050,105,1052,1051,1054,1053,1059,1055,1060,106,1062,1061,1064,1063,1066,1065,1068,1067,107,1069,1074,1070,1076,1075,1078,1077,108,1079,1081,1080,1083,1082,1085,1084,109,1086,1091,1090,1093,1092,1095,1094,1097,1096,1099,1098,110,11,1104,1100,1106,1105,1108,1107,111,1109,1111,1110,1113,1112,1115,1114,1136,1116,1138,1137,114,1139,1141,1140,1143,1142,1145,1144,1147,1146,1151,1148,1153,1152,1155,1154,1157,1156,1159,1158,1160,116,1162,1161,1167,1163,1169,1168,1170,117,1172,1171,1174,1173,1176,1175,1178,1177,1182,1179,1184,1183,1186,1185,1188,1187,119,1189,1191,1190,1193,1192,1198,1194,12,1199,1200,120,1202,1201,1204,1203,1206,1205,1208,1207,1212,1209,1214,1213,1216,1215,1218,1217,122,1219,1221,1220,1223,1222,1228,1224,123,1229,1231,1230,1233,1232,1235,1234,1237,1236,1239,1238,1240,124,1243,1241,1245,1244,1247,1246,1249,1248,1250,125,1252,1251,1254,1253,1256,1255,1259,1257,1260,126,1262,1261,1264,1263,1266,1265,1268,1267,127,1269,1271,1270,1273,1272,1275,1274,1277,1276,1279,1278,1297,1296,1299,1298,130,13,1301,1300,1303,1302,1305,1304,1307,1306,1309,1308,128,1258,1282,2278,1311,1310,1313,1312,1315,1314,1317,1316,1319,1318,1320,132,1322,1321,1324,1323,1327,1326,1329,1328,1330,133,1332,1331,1334,1333,1336,1335,1338,1337,134,1339,1342,1341,1344,1343,1346,1345,1348,1347,135,1349,1351,1350,1353,1352,1355,1354,1358,1357,136,1359,1361,1360,1363,1362,1365,1364,1367,1366,1369,1368,1370,137,1373,1372,1375,1374,1377,1376,1379,1378,1380,138,1382,1381,1384,1383,1386,1385,1389,1388,1390,139,1392,1391,1394,1393,1396,1395,1398,1397,14,1399,1400,140,1403,1402,1405,1404,1407,1406,1409,1408,1410,141,1412,1411,1414,1413,1416,1415,1418,1417,142,1419,1421,1420,1423,1422,1425,1424,1427,1426,1429,1428,1447,1446,1449,1448,1450,145,1452,1451,1454,1453,1456,1455,1458,1457,146,1459,1462,1461,1464,1463,1466,1465,1468,1467,147,1469,1471,1470,1473,1472,1475,1474,1478,1477,148,1479,1481,1480,1483,1482,1485,1484,1487,1486,1489,1488,1490,149,1493,1492,1495,1494,1497,1496,1499,1498,150,15,1501,1500,1503,1502,1505,1504,1508,1507,151,1509,1511,1510,1513,1512,1515,1514,1517,1516,1519,1518,1520,152,1523,1522,1525,1524,1527,1526,1529,1528,1530,153,1532,1531,1534,1533,1536,1535,1539,1538,1540,154,1542,1541,1544,1543,1546,1545,1548,1547,155,1549,1551,1550,1554,1553,1556,1555,1558,1557,156,1559,1561,1560,1563,1562,1565,1564,1567,1566,157,1569,1571,1570,1573,1572,1575,1574,1577,1576,1579,1578,1580,158,1582,1581,1584,1583,1586,1585,1588,1587,1521,1589,1552,1537,159,1568,1592,1608,1607,161,1609,1611,1610,1613,1612,1615,1614,1617,1616,1619,1618,1621,1620,1623,1622,1625,1624,1627,1626,1629,1628,1630,163,1632,1631,1634,1633,1637,1636,1639,1638,1640,164,1642,1641,1644,1643,1646,1645,1648,1647,165,1649,1652,1651,1654,1653,1656,1655,1658,1657,166,1659,1661,1660,1663,1662,1665,1664,1668,1667,167,1669,1671,1670,1673,1672,1675,1674,1677,1676,1679,1678,1680,168,1683,1682,1685,1684,1687,1686,1689,1688,1690,169,1692,1691,1694,1693,1696,1695,1699,1698,170,17,1701,1700,1703,1702,1705,1704,1707,1706,1709,1708,1710,171,1713,1712,1715,1714,1717,1716,1719,1718,1720,172,1722,1721,1724,1723,1726,1725,1728,1727,173,1729,1731,1730,1733,1732,1735,1734,1737,1736,1739,1738,1757,1756,1759,1758,1760,176,1762,1761,1764,1763,1766,1765,1768,1767,177,1769,1772,1771,1774,1773,1776,1775,1778,1777,178,1779,1781,1780,1783,1782,1785,1784,1788,1787,179,1789,1791,1790,1793,1792,1795,1794,1797,1796,1799,1798,180,18,1802,1801,1804,1803,1806,1805,1808,1807,181,1809,1811,1810,1813,1812,1815,1814,1818,1817,182,1819,1821,1820,1823,1822,1825,1824,1827,1826,1829,1828,1830,183,1833,1832,1835,1834,1837,1836,1839,1838,1840,184,1842,1841,1844,1843,1846,1845,1849,1848,1850,185,1852,1851,1854,1853,1856,1855,1858,1857,186,1859,1861,1860,1864,1863,1866,1865,1868,1867,187,1869,1871,1870,1873,1872,1875,1874,1877,1876,1800,1786,1831,1816,1862,1847,19,1878,188,1879,1881,1880,1883,1882,1885,1884,1887,1886,1889,1888,1890,189,1892,1891,1894,1893,1896,1895,1898,1897,1915,1899,1917,1916,1919,1918,1920,192,1922,1921,1924,1923,1926,1925,1928,1927,1930,1929,1932,1931,1934,1933,1936,1935,1938,1937,194,1939,1941,1940,1943,1942,1946,1944,1948,1947,195,1949,1951,1950,1953,1952,1955,1954,1957,1956,1959,1958,1961,196,1963,1962,1965,1964,1967,1966,1969,1968,1970,197,1972,1971,1974,1973,1977,1975,1979,1978,1980,198,1982,1981,1984,1983,1986,1985,1988,1987,199,1989,1992,1990,1994,1993,1996,1995,1998,1997,2,1999,200,20,2001,2000,2003,2002,2006,2004,2008,2007,201,2009,2011,2010,2013,2012,2015,2014,2017,2016,2019,2018,2021,202,2023,2022,2025,2024,2027,2026,2029,2028,2030,203,2032,2031,2034,2033,2036,2035,2038,2037,2140,2039,2142,2141,2144,2143,2146,2145,2148,2147,215,2149,2151,2150,2153,2152,2056,2154,2058,2057,206,2059,2061,2060,2063,2062,2065,2064,2067,2066,2069,2068,2071,207,2073,2072,2075,2074,2077,2076,2079,2078,2080,208,2082,2081,2084,2083,2087,2085,2089,2088,2090,209,2092,2091,2094,2093,2096,2095,2098,2097,21,2099,2101,210,2103,2102,2105,2104,2107,2106,2109,2108,2110,211,2112,2111,2114,2113,2117,2115,2119,2118,2120,212,2122,2121,2124,2123,2126,2125,2128,2127,213,2129,2131,2130,2133,2132,2135,2134,2137,2136,2139,2138,2157,2156,2159,2158,2160,216,2171,2162,2161,2164,2163,2166,2165,2168,2167,217,2169,2174,204,2042,2173,2176,2175,2178,2177,218,2179,2181,2180,2183,2182,2185,2184,2188,2187,219,2189,2191,2190,2193,2192,2195,2194,2197,2196,2199,2198,220,22,2202,2201,2204,2203,2206,2205,2208,2207,221,2209,2211,2210,2213,2212,2215,2214,2218,2217,222,2219,2221,2220,2223,2222,2225,2224,2227,2226,2229,2228,2230,223,2233,2232,2235,2234,2237,2236,2239,2238,2240,224,2242,2241,2244,2243,2246,2245,2249,2248,2250,225,2252,2251,2254,2253,2256,2255,2258,2257,226,2259,2261,2260,2264,2263,2266,2265,2268,2267,227,2269,2271,2270,2273,2272,2275,2274,2277,2276,228,2279,2281,2280,2283,2282,2285,2284,2287,2286,2289,2288,2290,229,2292,2291,2294,2293,2296,2295,2298,2297,2315,2299,2317,2316,2319,2318,2320,232,2322,2321,2324,2323,2326,2325,2328,2327,2330,2329,2332,2331,2334,2333,2336,2335,2338,2337,234,2339,2341,2340,2343,2342,2346,2344,2348,2347,235,2349,2351,2350,2353,2352,2355,2354,2357,2356,2359,2358,2361,236,2363,2362,2365,2364,2367,2366,2369,2368,2370,237,2372,2371,2374,2373,2377,2375,2379,2378,2380,238,2382,2381,2384,2383,2386,2385,2388,2387,239,2389,2392,2390,2394,2393,2396,2395,2398,2397,24,2399,2400,240,2402,2401,2404,2403,2407,2405,2409,2408,2410,241,2412,2411,2414,2413,2416,2415,2418,2417,242,2419,2422,2420,2424,2423,2426,2425,2428,2427,243,2429,2431,2430,2433,2432,2435,2434,2437,2436,2439,2438,2440,244,2442,2441,2444,2443,2446,2445,2448,2447,23,2449,2301,2469,2468,2470,247,2472,2471,2474,2473,2476,2475,2478,2477,248,2479,2482,2481,2484,2483,2486,2485,2488,2487,249,2489,2491,2490,2493,2492,2495,2494,2498,2497,25,2499,2500,250,2502,2501,2504,2503,2506,2505,2508,2507,251,2509,2512,2511,2514,2513,2516,2515,2518,2517,252,2519,2521,2520,2523,2522,2525,2524,2528,2527,253,2529,2531,2530,2533,2532,2535,2534,2537,2536,2539,2538,2540,254,2543,2542,2545,2544,2547,2546,2549,2548,2550,255,2552,2551,2554,2553,2556,2555,2559,2558,2560,256,2562,2561,2564,2563,2566,2565,2568,2567,257,2569,2571,2570,2574,2573,2576,2575,2578,2577,258,2579,2581,2580,2583,2582,2585,2584,2587,2586,2589,2588,2590,259,2592,2591,2594,2593,2596,2595,2598,2597,26,2599,2617,2616,2619,2618,2620,262,2622,2621,2624,2623,2626,2625,2628,2627,263,2629,2632,2631,2634,2633,2636,2635,2638,2637,264,2639,2641,2640,2643,2642,2645,2644,2648,2647,265,2649,2651,2650,2653,2652,2655,2654,2657,2656,2659,2658,2660,266,2663,2662,2665,2664,2667,2666,2669,2668,2670,267,2672,2671,2674,2673,2676,2675,2679,2678,2680,268,2682,2681,2684,2683,2686,2685,2688,2687,269,2689,2691,2690,2694,2693,2696,2695,2698,2697,27,2699,2700,270,2702,2701,2704,2703,2706,2705,2709,2708,2710,271,2712,2711,2714,2713,2716,2715,2718,2717,272,2719,2721,2720,2724,2723,2726,2725,2728,2727,273,2729,2731,2730,2739,2733,2732,2735,2734,2737,2736,2572,2557,2741,260,2602,2740,2743,2742,2745,2744,2747,2746,2749,2748,2750,275,2752,2751,2754,2753,2756,2755,2758,2757,2776,2759,2778,2777,278,2779,2781,2780,2783,2782,2785,2784,2787,2786,2789,2788,2791,279,2793,2792,2795,2794,2797,2796,2799,2798,280,28,2801,2800,2803,2802,2806,2804,2808,2807,281,2809,2811,2810,2813,2812,2815,2814,2817,2816,2819,2818,2821,282,2823,2822,2825,2824,2827,2826,2829,2828,2830,283,2832,2831,2834,2833,2837,2835,2839,2838,2840,284,2842,2841,2844,2843,2846,2845,2848,2847,285,2849,2852,2850,2854,2853,2856,2855,2858,2857,286,2859,2861,2860,2863,2862,2865,2864,2868,2866,287,2869,2871,2870,2873,2872,2875,2874,2877,2876,2879,2878,2880,288,2883,2881,2885,2884,2887,2886,2889,2888,2890,289,2892,2891,2894,2893,2896,2895,2898,2897,29,2899,2900,290,2902,2901,2904,2903,2906,2905,2908,2907,2926,2909,2928,2927,293,2929,2931,2930,2933,2932,2935,2934,2937,2936,2939,2938,2941,294,2943,2942,2945,2944,2947,2946,2949,2948,2950,295,2952,2951,2954,2953,2957,2955,2959,2958,2960,296,2962,2961,2964,2963,2966,2965,2968,2967,297,2969,2972,2970,2974,2973,2976,2975,2978,2977,298,2979,2981,2980,2983,2982,2985,2984,2988,2986,299,2989,2991,2990,2993,2992,2995,2994,2997,2996,2999,2998,30,3,3001,300,3003,3002,3005,3004,3007,3006,3009,3008,3010,301,3012,3011,3014,3013,3017,3015,3019,3018,3020,302,3022,3021,3024,3023,3026,3025,3028,3027,303,3029,2820,3030,2851,2836,2882,2867,3032,291,3034,3033,3036,3035,3038,3037,304,3039,3041,3040,3043,3042,3045,3044,3047,3046,3049,3048,3050,305,3052,3051,3054,3053,3056,3055,3058,3057,3076,3059,3078,3077,308,3079,3081,3080,3083,3082,3085,3084,3087,3086,3089,3088,3091,309,3093,3092,3095,3094,3097,3096,3099,3098,310,31,3101,3100,3103,3102,3106,3104,3108,3107,311,3109,3111,3110,3113,3112,3115,3114,3117,3116,3119,3118,3121,312,3123,3122,3125,3124,3127,3126,3129,3128,3130,313,3132,3131,3134,3133,3137,3135,3139,3138,3140,314,3142,3141,3144,3143,3146,3145,3148,3147,315,3149,3152,3150,3154,3153,3156,3155,3158,3157,316,3159,3161,3160,3163,3162,3165,3164,3168,3166,317,3169,3171,3170,3173,3172,3175,3174,3177,3176,3179,3178,3180,318,3183,3181,3185,3184,3187,3186,3189,3188,3190,319,3192,3191,3194,3193,3196,3195,3198,3197,32,3199,3200,320,3202,3201,3204,3203,3206,3205,3208,3207,3226,3209,3228,3227,323,3229,3231,3230,3233,3232,3235,3234,3237,3236,3239,3238,3241,324,3243,3242,3245,3244,3247,3246,3249,3248,3250,325,3252,3251,3254,3253,3257,3255,3259,3258,3260,326,3262,3261,3264,3263,3266,3265,3268,3267,327,3269,3272,3270,3274,3273,3276,3275,3278,3277,328,3279,3281,3280,3283,3282,3285,3284,3288,3286,329,3289,3291,3290,3293,3292,3295,3294,3297,3296,3299,3298,330,33,3302,3300,3304,3303,3306,3305,3308,3307,331,3309,3311,3310,3313,3312,3315,3225,3314,3318,3316,332,3319,3321,3320,3323,3322,3325,3324,3327,3326,3329,3328,3330,333,3090,3331,3120,3105,3151,3136,3182,3167,321,3212,3333,3335,3334,3337,3336,3339,3338,3340,334,3342,3341,3344,3343,3346,3345,3349,3347,3350,335,3352,3351,3354,3353,3356,3355,3358,3357,336,3359,3361,3360,3363,3362,3365,3364,3367,3366,3369,3368,3387,3386,3389,3388,3390,339,3392,3391,3394,3393,3396,3395,3398,3397,34,3399,3401,3400,3403,3402,3405,3404,3407,3406,3409,3408,3410,341,3412,3411,3414,3413,3417,3416,3419,3418,3420,342,3422,3421,3424,3423,3426,3425,3428,3427,343,3429,3432,3431,3434,3433,3436,3435,3438,3437,344,3439,3441,3440,3443,3442,3445,3444,3448,3447,345,3449,3451,3450,3453,3452,3455,3454,3457,3456,3459,3458,3460,346,3463,3462,3465,3464,3467,3466,3469,3468,3470,347,3472,3471,3474,3473,3476,3475,3479,3478,3480,348,3482,3481,3484,3483,3486,3485,3488,3487,349,3489,3491,3490,3494,3493,3496,3495,3498,3497,35,3499,3500,350,3502,3501,3504,3503,3506,3505,3508,3507,351,3509,3511,3510,3513,3512,3515,3514,3517,3516,3519,3518,3537,3536,3539,3538,3540,354,3542,3541,3544,3543,3546,3545,3548,3547,3552,3551,3554,3553,3556,3555,3558,3557,356,3559,3561,3560,3563,3562,3568,3567,357,3569,3571,3570,3573,3572,3575,3574,3577,3576,3579,3578,3583,3582,3585,3584,3587,3586,3589,3588,3590,359,3592,3591,3594,3593,3599,3598,360,36,3601,3600,3603,3602,3605,3604,3607,3606,3609,3608,3613,3612,3615,3614,3617,3616,3619,3618,3620,362,3622,3621,3624,3623,3629,3628,3630,363,3632,3631,3634,3633,3636,3635,3638,3637,364,3639,3644,3643,3646,3645,3648,3647,365,3649,3651,3650,3653,3652,3655,3654,366,3659,3661,3660,3663,3662,3665,3664,3667,3666,3669,3668,3670,367,3741,3740,3743,3742,3745,3744,3747,3746,3749,3748,3750,375,3752,3751,3697,3696,3699,3698,370,37,3701,3700,3703,3702,3705,3704,3707,3706,3711,3710,3713,3712,3715,3714,3717,3716,3719,3718,3720,372,3722,3721,3756,3723,3758,3757,376,3759,3761,3760,3763,3762,3765,3764,3767,3766,3771,3768,3773,3772,3775,3774,3777,3776,3779,3778,3780,378,3782,3781,3784,3783,3787,3785,3789,3788,3790,379,3792,3791,3794,3793,3796,3795,3798,3797,38,3799,3801,380,3803,3802,3805,3804,3807,3806,3809,3808,3810,381,3812,3811,3814,3813,3817,3815,3819,3818,3820,382,3822,3821,3824,3823,3826,3825,3838,3828,3827,383,3829,3851,3800,3830,3831,3816,3862,3847,39,3878,390,387,3880,388,3882,3881,3884,3883,3915,3885,3917,3916,3919,3918,3920,392,3922,3921,3924,3923,3926,3925,3928,3927,3930,3929,3932,3931,3934,3933,3936,3935,3938,3937,394,3939,3941,3940,3943,3942,3946,3944,3948,3947,395,3949,3951,3950,3953,3952,3955,3954,3957,3956,3959,3958,3961,396,3963,3962,3965,3964,3967,3966,3969,3968,3970,397,3972,3971,3974,3973,3977,3975,3979,3978,3980,398,3982,3981,3984,3983,3986,3985,3988,3987,399,3989,3992,3990,3994,3993,3996,3995,3998,3997,4,3999,400,40,4001,4000,4003,4002,4006,4004,4008,4007,401,4009,4011,4010,4013,4012,4015,4014,4017,4016,4019,4018,4021,402,4023,4022,4025,4024,4027,4026,4029,4028,4030,403,4032,4031,4034,4033,4036,4035,4038,4037,4056,4039,4058,4057,406,4059,4061,4060,4063,4062,4065,4064,4067,4066,4069,4068,4071,407,4073,4072,4075,4074,4077,4076,4079,4078,4080,408,4082,4081,4084,4083,4087,4085,4089,4088,4090,409,4092,4091,4094,4093,4096,4095,4098,4097,41,4099,4101,410,4103,4102,4105,4104,4107,4106,4109,4108,4110,411,4112,4111,4114,4113,4120,4119,4115,4042,4118,412,4122,4121,4124,4123,4126,4125,4128,4127,413,4129,4132,4130,4134,4133,4136,4135,4138,4137,414,4139,4141,4140,4143,4142,4145,4144,4148,4146,415,4149,4151,4150,4153,4152,4155,4154,4157,4156,4159,4158,4160,416,4163,4161,4165,4164,4167,4166,4169,4168,4170,417,4172,4171,4174,4173,4176,4175,4179,4177,4180,418,4182,4181,4184,4183,4186,4185,4188,4187,419,4189,4191,4190,4193,4192,4195,4194,4197,4196,4199,4198,4216,4215,4218,4217,422,4219,4221,4220,4223,4222,4225,4224,4227,4226,4229,4228,4231,4230,4233,4232,4235,4234,4237,4236,4239,4238,4240,424,4242,4241,4244,4243,4247,4246,4249,4248,4250,425,4252,4251,4254,4253,4256,4255,4258,4257,426,4259,4262,4261,4264,4263,4266,4265,4268,4267,427,4269,4271,4270,4273,4272,4275,4274,4278,4277,428,4279,4281,4280,4283,4282,4285,4284,4287,4286,4289,4288,4290,429,4293,4292,4295,4294,4297,4296,4299,4298,430,43,4301,4300,4303,4302,4305,4304,4308,4307,431,4309,4311,4310,4313,4312,4315,4314,4317,4316,4319,4318,4320,432,4323,4322,4325,4324,4327,4326,4329,4328,4330,433,4332,4331,4334,4333,4336,4335,4338,4337,434,4339,4341,4340,4343,4342,4345,4344,4347,4346,4349,4348,4367,4366,4369,4368,4370,437,4372,4371,4374,4373,4376,4375,4378,4377,438,4379,4382,4381,4384,4383,4386,4385,4388,4387,439,4389,4391,4390,4393,4392,4395,4394,4398,4397,44,4399,4400,440,4402,4401,4404,4403,4406,4405,4408,4407,441,4409,4321,4416,4306,4412,435,4352,4415,4418,4417,442,4419,4421,4420,4423,4422,4425,4424,4428,4427,443,4429,4431,4430,4433,4432,4435,4434,4437,4436,4439,4438,4440,444,4443,4442,4445,4444,4447,4446,4449,4448,4450,445,4452,4451,4454,4453,4456,4455,4459,4458,4460,446,4462,4461,4464,4463,4466,4465,4468,4467,447,4469,4471,4470,4474,4473,4476,4475,4478,4477,448,4479,4481,4480,4483,4482,4485,4484,4487,4486,4489,4488,4490,449,4492,4491,4494,4493,4496,4495,4498,4497,45,4499,4517,4516,4519,4518,4520,452,4522,4521,4524,4523,4526,4525,4528,4527,453,4529,4532,4531,4534,4533,4536,4535,4538,4537,454,4539,4541,4540,4543,4542,4545,4544,4548,4547,455,4549,4551,4550,4553,4552,4555,4554,4557,4556,4559,4558,4560,456,4563,4562,4565,4564,4567,4566,4569,4568,4570,457,4572,4571,4574,4573,4576,4575,4579,4578,4580,458,4582,4581,4584,4583,4586,4585,4588,4587,459,4589,4591,4590,4594,4593,4596,4595,4598,4597,46,4599,4600,460,4602,4601,4604,4603,4606,4605,4609,4608,4610,461,4612,4611,4614,4613,4616,4615,4618,4617,462,4619,4621,4620,4624,4623,4626,4625,4628,4627,463,4629,4631,4630,4633,4632,4635,4634,4637,4636,464,4639,4641,4640,4643,4642,4645,4644,4647,4646,4649,4648,4650,465,4652,4651,4654,4653,4656,4655,4658,4657,4676,4659,4678,4677,468,4679,4681,4680,4683,4682,4685,4684,4687,4686,4689,4688,4691,469,4693,4692,4695,4694,4697,4696,4699,4698,470,47,4701,4700,4703,4702,4577,4704,4607,4710,4592,4638,4622,4707,466,4661,471,4712,4711,4714,4713,4716,4715,4718,4717,472,4719,4722,4721,4724,4723,4726,4725,4728,4727,473,4729,4731,4730,4733,4732,4735,4734,4738,4737,474,4739,4741,4740,4743,4742,4745,4744,4747,4746,4749,4748,4750,475,4753,4752,4755,4754,4757,4756,4759,4758,4760,476,4762,4761,4764,4763,4766,4765,4769,4768,4770,477,4772,4771,4774,4773,4776,4775,4778,4777,478,4779,4781,4780,4784,4783,4786,4785,4788,4787,479,4789,4791,4790,4793,4792,4795,4794,4797,4796,4799,4798,480,48,4801,4800,4803,4802,4805,4804,4807,4806,4809,4808,4827,4826,4829,4828,4830,483,4832,4831,4834,4833,4836,4835,4838,4837,484,4839,4842,4841,4844,4843,4846,4845,4848,4847,485,4849,4851,4850,4853,4852,4855,4854,4858,4857,486,4859,4861,4860,4863,4862,4865,4864,4867,4866,4869,4868,4870,487,4873,4872,4875,4874,4877,4876,4879,4878,4880,488,4882,4881,4884,4883,4886,4885,4889,4888,4890,489,4892,4891,4894,4893,4896,4895,4898,4897,49,4899,4900,490,4903,4902,4905,4904,4907,4906,4909,4908,4910,491,4912,4911,4914,4913,4916,4915,4919,4918,4920,492,4922,4921,4924,4923,4926,4925,4928,4927,493,4929,4931,4930,4934,4933,4936,4935,4938,4937,494,4939,4941,4940,4943,4942,4945,4944,4947,4946,495,4949,4951,4950,4953,4952,4955,4954,4957,4956,4959,4958,4960,496,4962,4961,4964,4963,4966,4965,4968,4967,4986,4969,4988,4987,499,4989,4991,4990,4993,4992,4995,4994,4997,4996,4999,4998,4840,5,4871,5001,4856,4901,4887,4932,4917,497,4948,4971,5002,5005,5004,5007,5006,5009,5008,5010,501,5012,5011,5015,5013,5017,5016,5019,5018,5020,502,5022,5021,5024,5023,5026,5025,5028,5027,5030,5029,5032,5031,5034,5033,5036,5035,5038,5037,504,5039,5041,5040,5043,5042,5046,5044,5048,5047,505,5049,5051,5050,5053,5052,5055,5054,5057,5056,5059,5058,5061,506,5063,5062,5065,5064,5067,5066,5069,5068,5070,507,5072,5071,5074,5073,5077,5075,5079,5078,5080,508,5082,5081,5084,5083,5086,5085,5088,5087,509,5089,5092,5090,5094,5093,5096,5095,5098,5097,51,5099,5100,510,5102,5101,5104,5103,5106,5105,5108,5107,5240,5109,5242,5241,5244,5243,5246,5245,5248,5247,525,5249,5251,5250,5253,5252,5126,5254,5128,5127,513,5129,5131,5130,5133,5132,5135,5134,5137,5136,5139,5138,5141,514,5143,5142,5145,5144,5147,5146,5149,5148,5150,515,5152,5151,5154,5153,5157,5155,5159,5158,5160,516,5162,5161,5164,5163,5166,5165,5168,5167,517,5169,5172,5170,5174,5173,5176,5175,5178,5177,518,5179,5181,5180,5183,5182,5185,5184,5188,5186,519,5189,5191,5190,5193,5192,5195,5194,5197,5196,5199,5198,520,52,5202,5200,5204,5203,5206,5205,5208,5207,521,5209,5211,5210,5213,5212,5215,5214,5218,5216,522,5219,5221,5220,5223,5222,5225,5224,5227,5226,5229,5228,5230,523,5232,5231,5234,5233,5236,5235,5238,5237,5256,5239,5258,5257,526,5271,5259,5261,5260,5263,5262,5265,5264,5267,5266,5269,5268,5091,527,5274,511,5112,5273,5276,5275,5278,5277,528,5279,5281,5280,5283,5282,5285,5284,5288,5287,529,5289,5291,5290,5293,5292,5295,5294,5297,5296,5299,5298,530,53,5302,5301,5304,5303,5306,5305,5308,5307,531,5309,5311,5310,5313,5312,5315,5314,5318,5317,532,5319,5321,5320,5323,5322,5325,5324,5327,5326,5329,5328,5330,533,5333,5332,5335,5334,5337,5336,5339,5338,5340,534,5342,5341,5344,5343,5346,5345,5349,5348,5350,535,5352,5351,5354,5353,5356,5355,5358,5357,536,5359,5361,5360,5364,5363,5366,5365,5368,5367,537,5369,5371,5370,5373,5372,5375,5374,5377,5376,538,5379,5381,5380,5383,5382,5385,5384,5387,5386,5389,5388,5390,539,5392,5391,5394,5393,5396,5395,5398,5397,5415,5399,5417,5416,5419,5418,5420,542,5422,5421,5424,5423,5426,5425,5428,5427,5430,5429,5432,5431,5434,5433,5436,5435,5438,5437,544,5439,5441,5440,5443,5442,5446,5444,5448,5447,545,5449,5451,5450,5453,5452,5455,5454,5457,5456,5459,5458,5461,546,5463,5462,5465,5464,5467,5466,5469,5468,5470,547,5472,5471,5474,5473,5477,5475,5479,5478]

for data_name in data_file_list:
    if not (int(data_name[19:-5]) in include_files):
        continue
    print("Processing", data_name)
    data_file=open(data_name,'r')
    df_lines=data_file.readlines()
    df_lines[27]='15 55 5 /\n'
    df_lines[78]='4125*40 /\n'
    df_lines[80]='4125*20 /\n'
    df_lines[82]='4125*2 /\n'

    parts=df_lines[215].split()
    parts[2]=str(int(max(1,int(parts[2])/2)))
    parts[3]=str(int(max(1,int(parts[3])/2)))
    df_lines[215] = '  '.join(parts)+'\n'

    parts=df_lines[216].split()
    parts[2]=str(int(max(1,int(parts[2])/2)))
    parts[3]=str(int(max(1,int(parts[3])/2)))
    df_lines[216] = '  '.join(parts)+'\n'

    parts=df_lines[220].split()
    parts[1]=str(int(max(1,int(parts[1])/2)))
    parts[2]=str(int(max(1,int(parts[2])/2)))
    df_lines[220] = '  '.join(parts)+'\n'

    parts=df_lines[221].split()
    parts[1]=str(int(max(1,int(parts[1])/2)))
    parts[2]=str(int(max(1,int(parts[2])/2)))
    df_lines[221] = '  '.join(parts)+'\n'

    new_data_name = 'up_same_geo/'+data_name.split('/')[1]
    new_data_file = open(new_data_name, 'w')
    new_data_file.writelines(df_lines)
    new_data_file.close()

    data_file.close()
