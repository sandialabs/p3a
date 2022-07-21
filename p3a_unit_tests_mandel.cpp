#include "gtest/gtest.h"
#include "p3a_mandel3x6.hpp"
#include "p3a_mandel6x3.hpp"
#include "p3a_mandel6x6.hpp"

using Y = double;

Y abs_err = Y(5.) * p3a::epsilon_value<Y>();

struct TestData {

  Y const r1 =  0.2946950525769927;
  p3a::vector3<Y> v;
  p3a::vector3<Y> vp;
  p3a::mandel6x1<Y> V;
  p3a::mandel6x1<Y> VinvX;
  p3a::mandel6x1<Y> ident;
  p3a::symmetric3x3<Y> Vsymm;
  p3a::diagonal3x3<Y> Vdiag;
  p3a::mandel6x1<Y> Vp;
  p3a::mandel6x1<Y> VpinvX;
  p3a::matrix3x3<Y> TV;
  p3a::static_matrix<Y,3,3> TVstatic;
  p3a::mandel6x6<Y> C;
  p3a::static_matrix<Y,6,6> Cstatic;
  p3a::mandel6x6<Y> CinvX;
  p3a::mandel6x6<Y> Ident;
  p3a::mandel6x6<Y> Cp;
  p3a::mandel6x6<Y> CpinvX;
  p3a::mandel3x6<Y> e36;
  p3a::mandel3x6<Y> e36invX;
  p3a::mandel3x6<Y> e36p;  
  p3a::mandel3x6<Y> e36pinvX;
  p3a::mandel6x3<Y> e63;
  p3a::mandel6x3<Y> e63invX;
  p3a::mandel6x3<Y> e63p;
  p3a::mandel6x3<Y> e63pinvX;
  
  TestData() : v(0.5403153418505242, 0.1916615978226233, 0.7913471057605709), 
               vp(0.2588658912519261, 0.9328687907855321, 0.2121274165250259), 
               V(0.3458750937599743, 0.1560730533788416, 0.6018325296947370,
                 1.3672391918240188, 0.0056991146342982, 1.1552653743154688,false),
               ident(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, false), 
               Vsymm(0.3458750937599743, 0.8168959802484831, 0.0040298826046717,
                                         0.1560730533788416, 0.9667841040427784,
                                                             0.6018325296947370),
               Vdiag(0.3458750937599743, 0.8168959802484831, 0.0040298826046717),
               Vp(0.0332809212161841, 0.0487815715174135, 0.8094624540252358,
                  0.4008775383705890, 0.6915662506016785, 0.1162162432201077,false),
               TV(0.3458750937599743, 0.8168959802484831, 0.0040298826046717,
                  0.8168959802484831, 0.1560730533788416, 0.9667841040427785,
                  0.0040298826046717, 0.9667841040427785, 0.6018325296947370),
               C(0.4335153608215544, 0.0856007508096491, 0.6181864586228960, 1.0180326366236518, 0.9998149495390550, 1.3248978263217353,
                 0.4327952022077987, 0.3742135173559312, 0.1080020759711545, 0.8147725263605445, 0.9048931279343728, 1.2476465544083184,
                 0.1619267043337781, 0.8255468602509465, 0.4823784238849611, 0.6837503075397613, 0.0784973317433504, 0.8368479998334433,
                 0.0112200396694362, 0.8194354458300377, 1.1074040005476036, 0.6118023983888154, 0.4369542629777867, 0.6474343145633832,
                 0.0836906759361557, 0.9338502039452401, 0.9911298229891556, 1.6059479559586896, 1.9166663623233735, 0.8109702524757447,
                 0.0501618784358511, 0.9139519661627958, 0.4597365381087316, 1.0569116390136390, 0.6655431451182421, 0.9678972164817718,
                 false),
               Ident(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                     false),
               Cp(0.2107177965118975, 0.1575416472166756, 0.5687654031408809, 0.0187036991768127, 0.3377232536213498, 1.2061579849914903,
                  0.9690547680956271, 0.3015620428716760, 0.0117926801846069, 0.9799577916244374, 1.3212262938876842, 1.1437745297112247,
                  0.9477962697578962, 0.8362711354333037, 0.5573723018243313, 1.1469535452828015, 1.3018482448421571, 1.2345256143574863,
                  1.1283311601253478, 0.8511773708296858, 0.5606130265475940, 1.7722378395723914, 0.5637212302658510, 1.2310906790509168,
                  0.7107977300479361, 0.1201771916290927, 0.8985377565988617, 1.6007054847018209, 1.8945186334241886, 1.6472528162998632,
                  0.6135597538445192, 1.1831257314989769, 1.2978577062887620, 0.0659440885591793, 1.6164910199618434, 1.8004733085589755,
                  false),
               e36(0.8076127837790422, 0.0119054861420681, 0.3198164644230427, 0.1899714025547371, 0.0810795183410036, 0.3999599660633990,
                   0.0453167988352610, 0.2818248347311008, 0.1508671814551668, 0.2127405166893033, 0.6128683635764686, 0.4153444264017785,
                   0.5405046604793561, 0.4391783723249735, 0.4128631387354691, 0.5546654249524924, 0.6862523687819362, 0.6488819206134113,
                   false),
               e36p(0.1350855076155520, 0.0231767044698068, 0.9464219592050042, 0.1065057861528435, 0.2688248383393013, 0.1909939208122375,
                    0.6097426473273645, 0.7344489953830811, 0.8160894918005591, 0.3734452187287570, 0.2222233196071084, 0.5049930480412529,
                    0.2163593876325339, 0.0370067441280660, 0.3742486996899349, 0.3756620371052928, 0.2105298764336539, 0.3698923292958400,
                    false),
               e63(0.4759634421237341, 0.8867644229624673, 0.4135404273124733,
                   0.8568504365470173, 0.1472947329145138, 0.4477610978101461,
                   0.7307484706668289, 0.2502091219941188, 0.0617040225205491,
                   0.4119163419942556, 0.1499018857533562, 0.3357608808746134,
                   0.3481855982554204, 0.5219168123093268, 0.6767633687457478,
                   0.1958577356414437, 0.3181944873075155, 0.2417581438483243,
                   false),
               e63p(0.5608766530609716, 0.1521742176592169, 0.0109210634329266,
                    0.2873082726699938, 0.5245336001209888, 0.6563889052851136,
                    0.8040961321642845, 0.2438452443118183, 0.9532018807957614,
                    0.4882124674604895, 0.5228073932512954, 0.4459480468559615,
                    0.1874635649351754, 0.2063901066888893, 0.2852170092848386,
                    0.7058726767702264, 0.4070564297078077, 0.3017792316742351,
                    false) {
        VinvX = V;
        VinvX.invMandelXform();
        VpinvX = Vp;
        VpinvX.invMandelXform();

        CinvX = C;
        CinvX.invMandelXform();

        CpinvX = Cp;
        CpinvX.invMandelXform();

        e36invX = e36;
        e36invX.invMandelXform();

        e36pinvX = e36p;
        e36pinvX.invMandelXform();

        e63invX = e63;
        e63invX.invMandelXform();

        e63pinvX = e63p;
        e63pinvX.invMandelXform();

        TVstatic(0,0)=0.3458750937599743;
        TVstatic(0,1)=0.8168959802484831;
        TVstatic(0,2)=0.0040298826046717;
        TVstatic(1,0)=0.8168959802484831; 
        TVstatic(1,1)=0.1560730533788416; 
        TVstatic(1,2)=0.9667841040427785;
        TVstatic(2,0)=0.0040298826046717; 
        TVstatic(2,1)=0.9667841040427785;
        TVstatic(2,2)=0.6018325296947370;
        
        Cstatic(0,0)=0.4335153608215544;
        Cstatic(0,1)=0.0856007508096491;
        Cstatic(0,2)=0.6181864586228960; 
        Cstatic(0,3)=0.7198577808258045;
        Cstatic(0,4)=0.7069759307507516; 
        Cstatic(0,5)=0.9368442373714156; 
        Cstatic(1,0)=0.4327952022077987; 
        Cstatic(1,1)=0.3742135173559312; 
        Cstatic(1,2)=0.1080020759711545; 
        Cstatic(1,3)=0.5761311785140361; 
        Cstatic(1,4)=0.6398560670115011; 
        Cstatic(1,5)=0.8822193391461527;
        Cstatic(2,0)=0.1619267043337781;
        Cstatic(2,1)=0.8255468602509465; 
        Cstatic(2,2)=0.4823784238849611; 
        Cstatic(2,3)=0.4834844790997526; 
        Cstatic(2,4)=0.0555059955807731; 
        Cstatic(2,5)=0.5917408955046265;
        Cstatic(3,0)=0.0079337661354404; 
        Cstatic(3,1)=0.5794283604910414; 
        Cstatic(3,2)=0.7830528783003216; 
        Cstatic(3,3)=0.3059011991944077; 
        Cstatic(3,4)=0.2184771314888934; 
        Cstatic(3,5)=0.3237171572816916;
        Cstatic(4,0)=0.0591782444765415; 
        Cstatic(4,1)=0.6603318118221196; 
        Cstatic(4,2)=0.7008346188718544; 
        Cstatic(4,3)=0.8029739779793448; 
        Cstatic(4,4)=0.9583331811616868; 
        Cstatic(4,5)=0.4054851262378724;
        Cstatic(5,0)=0.0354698043990456;
        Cstatic(5,1)=0.6462616329524909; 
        Cstatic(5,2)=0.3250828236559117; 
        Cstatic(5,3)=0.5284558195068195; 
        Cstatic(5,4)=0.3327715725591210; 
        Cstatic(5,5)=0.4839486082408859;
    }
};

/**************************************************************************
 * Test Mandel Inverses:
 *   Many tests rely on the inverse being correct
 *************************************************************************/
TEST(mandel_tensors,LinAlg2ndOrderMandelVectorInverseV){

    TestData td;
    const auto & V = td.V;
    const auto & ident = td.ident;
    p3a::mandel6x1 T = inverse(V);
    p3a::mandel6x1 I = T*V;
    
    EXPECT_NEAR(ident.x1(),I.x1(),abs_err) << "I.x1";
    EXPECT_NEAR(ident.x2(),I.x2(),abs_err) << "I.x2";
    EXPECT_NEAR(ident.x3(),I.x3(),abs_err) << "I.x3";
    EXPECT_NEAR(ident.x4(),I.x4(),abs_err) << "I.x4";
    EXPECT_NEAR(ident.x5(),I.x5(),abs_err) << "I.x5";
    EXPECT_NEAR(ident.x6(),I.x6(),abs_err) << "I.x6";

}

TEST(mandel_tensors,LinAlg4thOrderMandelTensorInverseC){

    TestData td;
    const auto & C = td.C;
    const auto & Ident = td.Ident;

    p3a::mandel6x6 Ct = C;
    Ct.MandelXform();

    p3a::mandel6x6 St = inverse(Ct);
    St.invMandelXform();
    p3a::mandel6x6 S = C*St;

    EXPECT_NEAR(Ident.x11(),S.x11(),abs_err) << "S.x11()";
    EXPECT_NEAR(Ident.x12(),S.x12(),abs_err) << "S.x12()";
    EXPECT_NEAR(Ident.x13(),S.x13(),abs_err) << "S.x13()";
    EXPECT_NEAR(Ident.x14(),S.x14(),abs_err) << "S.x14()";
    EXPECT_NEAR(Ident.x15(),S.x15(),abs_err) << "S.x15()";
    EXPECT_NEAR(Ident.x16(),S.x16(),abs_err) << "S.x16()";
    EXPECT_NEAR(Ident.x21(),S.x21(),abs_err) << "S.x21()";
    EXPECT_NEAR(Ident.x22(),S.x22(),abs_err) << "S.x22()";
    EXPECT_NEAR(Ident.x23(),S.x23(),abs_err) << "S.x23()";
    EXPECT_NEAR(Ident.x24(),S.x24(),abs_err) << "S.x24()";
    EXPECT_NEAR(Ident.x25(),S.x25(),abs_err) << "S.x25()";
    EXPECT_NEAR(Ident.x26(),S.x26(),abs_err) << "S.x26()";
    EXPECT_NEAR(Ident.x31(),S.x31(),abs_err) << "S.x31()";
    EXPECT_NEAR(Ident.x32(),S.x32(),abs_err) << "S.x32()";
    EXPECT_NEAR(Ident.x33(),S.x33(),abs_err) << "S.x33()";
    EXPECT_NEAR(Ident.x34(),S.x34(),abs_err) << "S.x34()";
    EXPECT_NEAR(Ident.x35(),S.x35(),abs_err) << "S.x35()";
    EXPECT_NEAR(Ident.x36(),S.x36(),abs_err) << "S.x36()";
    EXPECT_NEAR(Ident.x41(),S.x41(),abs_err) << "S.x41()";
    EXPECT_NEAR(Ident.x42(),S.x42(),abs_err) << "S.x42()";
    EXPECT_NEAR(Ident.x43(),S.x43(),abs_err) << "S.x43()";
    EXPECT_NEAR(Ident.x44(),S.x44(),abs_err) << "S.x44()";
    EXPECT_NEAR(Ident.x45(),S.x45(),abs_err) << "S.x45()";
    EXPECT_NEAR(Ident.x46(),S.x46(),abs_err) << "S.x46()";
    EXPECT_NEAR(Ident.x51(),S.x51(),abs_err) << "S.x51()";
    EXPECT_NEAR(Ident.x52(),S.x52(),abs_err) << "S.x52()";
    EXPECT_NEAR(Ident.x53(),S.x53(),abs_err) << "S.x53()";
    EXPECT_NEAR(Ident.x54(),S.x54(),abs_err) << "S.x54()";
    EXPECT_NEAR(Ident.x55(),S.x55(),abs_err) << "S.x55()";
    EXPECT_NEAR(Ident.x56(),S.x56(),abs_err) << "S.x56()";
    EXPECT_NEAR(Ident.x61(),S.x61(),abs_err) << "S.x61()";
    EXPECT_NEAR(Ident.x62(),S.x62(),abs_err) << "S.x62()";
    EXPECT_NEAR(Ident.x63(),S.x63(),abs_err) << "S.x63()";
    EXPECT_NEAR(Ident.x64(),S.x64(),abs_err) << "S.x64()";
    EXPECT_NEAR(Ident.x65(),S.x65(),abs_err) << "S.x65()";
    EXPECT_NEAR(Ident.x66(),S.x66(),abs_err) << "S.x66()";
}

/**************************************************************************
 * Constructor tests for p3a::mandel6x1 (2nd-order tensors)
 *************************************************************************/
TEST(mandel_tensors,Construct2ndOrderVfromListxform){

    TestData td;
    const auto & V = td.V;

    p3a::mandel6x1 Vt = V; //V made from list initially

    EXPECT_FLOAT_EQ(V.x1(),Vt.x1()) << "V.x1";
    EXPECT_FLOAT_EQ(V.x2(),Vt.x2()) << "V.x2";
    EXPECT_FLOAT_EQ(V.x3(),Vt.x3()) << "V.x3";
    EXPECT_FLOAT_EQ(V.x4(),Vt.x4()) << "V.x4";
    EXPECT_FLOAT_EQ(V.x5(),Vt.x5()) << "V.x5";
    EXPECT_FLOAT_EQ(V.x6(),Vt.x6()) << "V.x6";
}

TEST(mandel_tensors,Construct2ndOrderVfromMatrix3x3Vxform){

    TestData td;
    const auto & V = td.V;
    const auto & TV = td.TV;
    p3a::mandel6x1 Vt = TV;

    EXPECT_FLOAT_EQ(V.x1(),Vt.x1()) << "V.x1";
    EXPECT_FLOAT_EQ(V.x2(),Vt.x2()) << "V.x2";
    EXPECT_FLOAT_EQ(V.x3(),Vt.x3()) << "V.x3";
    EXPECT_FLOAT_EQ(V.x4(),Vt.x4()) << "V.x4";
    EXPECT_FLOAT_EQ(V.x5(),Vt.x5()) << "V.x5";
    EXPECT_FLOAT_EQ(V.x6(),Vt.x6()) << "V.x6";
}

TEST(mandel_tensors,Construct2ndOrderVfromMatrix3x3V){

    TestData td;
    const auto & VinvX = td.VinvX;
    p3a::mandel6x1 Vt(td.TV);
    Vt.invMandelXform();

    EXPECT_FLOAT_EQ(VinvX.x1(),Vt.x1()) << "V.x1";
    EXPECT_FLOAT_EQ(VinvX.x2(),Vt.x2()) << "V.x2";
    EXPECT_FLOAT_EQ(VinvX.x3(),Vt.x3()) << "V.x3";
    EXPECT_FLOAT_EQ(VinvX.x4(),Vt.x4()) << "V.x4";
    EXPECT_FLOAT_EQ(VinvX.x5(),Vt.x5()) << "V.x5";
    EXPECT_FLOAT_EQ(VinvX.x6(),Vt.x6()) << "V.x6";
}

TEST(mandel_tensors,Construct2ndOrderVconvertfromStaticMatrixVxform){

    TestData td;
    const auto & V = td.V;
    const auto & TV = td.TVstatic;
    p3a::mandel6x1<Y> Vt = TV;

    EXPECT_FLOAT_EQ(V.x1(),Vt.x1()) << "V.x1";
    EXPECT_FLOAT_EQ(V.x2(),Vt.x2()) << "V.x2";
    EXPECT_FLOAT_EQ(V.x3(),Vt.x3()) << "V.x3";
    EXPECT_FLOAT_EQ(V.x4(),Vt.x4()) << "V.x4";
    EXPECT_FLOAT_EQ(V.x5(),Vt.x5()) << "V.x5";
    EXPECT_FLOAT_EQ(V.x6(),Vt.x6()) << "V.x6";
}

TEST(mandel_tensors,Construct2ndOrderVfromStaticMatrixVxform){

    TestData td;
    const auto & V = td.V;
    const auto & TV = td.TVstatic;
    p3a::mandel6x1 Vt(TV);

    EXPECT_FLOAT_EQ(V.x1(),Vt.x1()) << "V.x1";
    EXPECT_FLOAT_EQ(V.x2(),Vt.x2()) << "V.x2";
    EXPECT_FLOAT_EQ(V.x3(),Vt.x3()) << "V.x3";
    EXPECT_FLOAT_EQ(V.x4(),Vt.x4()) << "V.x4";
    EXPECT_FLOAT_EQ(V.x5(),Vt.x5()) << "V.x5";
    EXPECT_FLOAT_EQ(V.x6(),Vt.x6()) << "V.x6";
}

TEST(mandel_tensors,Construct2ndOrderVfromsymmetric3x3Vxform){

    TestData td;
    const auto & V = td.V;
    p3a::mandel6x1 Vt(td.Vsymm);    

    EXPECT_FLOAT_EQ(V.x1(),Vt.x1()) << "V.x1";
    EXPECT_FLOAT_EQ(V.x2(),Vt.x2()) << "V.x2";
    EXPECT_FLOAT_EQ(V.x3(),Vt.x3()) << "V.x3";
    EXPECT_FLOAT_EQ(V.x4(),Vt.x4()) << "V.x4";
    EXPECT_FLOAT_EQ(V.x5(),Vt.x5()) << "V.x5";
    EXPECT_FLOAT_EQ(V.x6(),Vt.x6()) << "V.x6";
}

TEST(mandel_tensors,Construct2ndOrderVfromsymmetric3x3V){

    TestData td;
    const auto & VinvX = td.VinvX;
    p3a::mandel6x1 Vt(td.Vsymm,false);    

    EXPECT_FLOAT_EQ(VinvX.x1(),Vt.x1()) << "V.x1";
    EXPECT_FLOAT_EQ(VinvX.x2(),Vt.x2()) << "V.x2";
    EXPECT_FLOAT_EQ(VinvX.x3(),Vt.x3()) << "V.x3";
    EXPECT_FLOAT_EQ(VinvX.x4(),Vt.x4()) << "V.x4";
    EXPECT_FLOAT_EQ(VinvX.x5(),Vt.x5()) << "V.x5";
    EXPECT_FLOAT_EQ(VinvX.x6(),Vt.x6()) << "V.x6";
}

/**************************************************************************
 * Basic operation tests for p3a::mandel6x1 (2nd-order tensors)
 *************************************************************************/
TEST(mandel_tensors,Basics2ndOrderVplusVp){

    TestData td;
    p3a::mandel6x1 Us = td.V + td.Vp;
    p3a::mandel6x1 U = Us-td.V-td.Vp;

    EXPECT_NEAR(0.,U.x1(),abs_err) << "U.x1";
    EXPECT_NEAR(0.,U.x2(),abs_err) << "U.x2";
    EXPECT_NEAR(0.,U.x3(),abs_err) << "U.x3";
    EXPECT_NEAR(0.,U.x4(),abs_err) << "U.x4";
    EXPECT_NEAR(0.,U.x5(),abs_err) << "U.x5";
    EXPECT_NEAR(0.,U.x6(),abs_err) << "U.x6";
}

TEST(mandel_tensors,Basics2ndOrderVplusequalVp){

    TestData td;
    p3a::mandel6x1 Us = td.V;
    Us += td.Vp;
    p3a::mandel6x1 U = Us-td.V-td.Vp;

    EXPECT_NEAR(0.,U.x1(),abs_err) << "U.x1";
    EXPECT_NEAR(0.,U.x2(),abs_err) << "U.x2";
    EXPECT_NEAR(0.,U.x3(),abs_err) << "U.x3";
    EXPECT_NEAR(0.,U.x4(),abs_err) << "U.x4";
    EXPECT_NEAR(0.,U.x5(),abs_err) << "U.x5";
    EXPECT_NEAR(0.,U.x6(),abs_err) << "U.x6";
}

TEST(mandel_tensors,Basics2ndOrderVxReal){

    TestData td;
    const auto & V = td.V;
    const auto & r1 = td.r1;
    p3a::mandel6x1 U = td.V; 
    U*=r1;

    EXPECT_FLOAT_EQ(V.x1()*r1,U.x1()) << "U.x1";
    EXPECT_FLOAT_EQ(V.x2()*r1,U.x2()) << "U.x2";
    EXPECT_FLOAT_EQ(V.x3()*r1,U.x3()) << "U.x3";
    EXPECT_FLOAT_EQ(V.x4()*r1,U.x4()) << "U.x4";
    EXPECT_FLOAT_EQ(V.x5()*r1,U.x5()) << "U.x5";
    EXPECT_FLOAT_EQ(V.x6()*r1,U.x6()) << "U.x6";
}

TEST(mandel_tensors,Basics2ndOrderVdivReal){

    TestData td;
    const auto & V = td.V;
    const auto & r1 = td.r1;
    p3a::mandel6x1 U = V;
    U /= r1;

    EXPECT_FLOAT_EQ(V.x1()/r1,U.x1()) << "U.x1";
    EXPECT_FLOAT_EQ(V.x2()/r1,U.x2()) << "U.x2";
    EXPECT_FLOAT_EQ(V.x3()/r1,U.x3()) << "U.x3";
    EXPECT_FLOAT_EQ(V.x4()/r1,U.x4()) << "U.x4";
    EXPECT_FLOAT_EQ(V.x5()/r1,U.x5()) << "U.x5";
    EXPECT_FLOAT_EQ(V.x6()/r1,U.x6()) << "U.x6";
}

TEST(mandel_tensors,Basics2ndOrderVminusVp){

    TestData td;
    const auto & V = td.V;
    const auto & Vp = td.Vp;
    p3a::mandel6x1 U = td.V - td.Vp;

    EXPECT_FLOAT_EQ(0.,V.x1()-Vp.x1()-U.x1()) << "U.x1";
    EXPECT_FLOAT_EQ(0.,V.x2()-Vp.x2()-U.x2()) << "U.x2";
    EXPECT_FLOAT_EQ(0.,V.x3()-Vp.x3()-U.x3()) << "U.x3";
    EXPECT_FLOAT_EQ(0.,V.x4()-Vp.x4()-U.x4()) << "U.x4";
    EXPECT_FLOAT_EQ(0.,V.x5()-Vp.x5()-U.x5()) << "U.x5";
    EXPECT_FLOAT_EQ(0.,V.x6()-Vp.x6()-U.x6()) << "U.x6";
}

TEST(mandel_tensors,Basics2ndOrderBinaryVmultReal){

    TestData td;
    const auto & V = td.V;
    const auto & r1 = td.r1;
    p3a::mandel6x1 U = V * r1;

    EXPECT_FLOAT_EQ(V.x1()*r1,U.x1()) << "U.x1";
    EXPECT_FLOAT_EQ(V.x2()*r1,U.x2()) << "U.x2";
    EXPECT_FLOAT_EQ(V.x3()*r1,U.x3()) << "U.x3";
    EXPECT_FLOAT_EQ(V.x4()*r1,U.x4()) << "U.x4";
    EXPECT_FLOAT_EQ(V.x5()*r1,U.x5()) << "U.x5";
    EXPECT_FLOAT_EQ(V.x6()*r1,U.x6()) << "U.x6";
}

TEST(mandel_tensors,Basics2ndOrderBinaryRealmultV){

    TestData td;
    const auto & V = td.V;
    const auto & r1 = td.r1;
    p3a::mandel6x1 U = r1 * V;

    EXPECT_FLOAT_EQ(V.x1()*r1,U.x1()) << "U.x1";
    EXPECT_FLOAT_EQ(V.x2()*r1,U.x2()) << "U.x2";
    EXPECT_FLOAT_EQ(V.x3()*r1,U.x3()) << "U.x3";
    EXPECT_FLOAT_EQ(V.x4()*r1,U.x4()) << "U.x4";
    EXPECT_FLOAT_EQ(V.x5()*r1,U.x5()) << "U.x5";
    EXPECT_FLOAT_EQ(V.x6()*r1,U.x6()) << "U.x6";
}

TEST(mandel_tensors,Basics2ndOrderBinaryVdivReal){

    TestData td;
    const auto & V = td.V;
    const auto & r1 = td.r1;
    p3a::mandel6x1 U = V / r1;

    EXPECT_FLOAT_EQ(V.x1()/r1,U.x1()) << "U.x1";
    EXPECT_FLOAT_EQ(V.x2()/r1,U.x2()) << "U.x2";
    EXPECT_FLOAT_EQ(V.x3()/r1,U.x3()) << "U.x3";
    EXPECT_FLOAT_EQ(V.x4()/r1,U.x4()) << "U.x4";
    EXPECT_FLOAT_EQ(V.x5()/r1,U.x5()) << "U.x5";
    EXPECT_FLOAT_EQ(V.x6()/r1,U.x6()) << "U.x6";
}

/**************************************************************************
 * Linear Alegbra operation tests for p3a::mandel6x1 (2nd-order tensors)
 *************************************************************************/
TEST(mandel_tensors,LinAlg2ndOrderInverseMandelXformV){

    TestData td;
    const auto & V = td.V;
    const auto & VinvX = td.VinvX;
    p3a::mandel6x1 T = V;
    T.invMandelXform();

    EXPECT_FLOAT_EQ(VinvX.x1(),T.x1()) << "T.x1";
    EXPECT_FLOAT_EQ(VinvX.x2(),T.x2()) << "T.x2";
    EXPECT_FLOAT_EQ(VinvX.x3(),T.x3()) << "T.x3";
    EXPECT_FLOAT_EQ(VinvX.x4(),T.x4()) << "T.x4";
    EXPECT_FLOAT_EQ(VinvX.x5(),T.x5()) << "T.x5";
    EXPECT_FLOAT_EQ(VinvX.x6(),T.x6()) << "T.x6";
}

TEST(mandel_tensors,LinAlg2ndOrderDeterminantV){

    TestData td;
    Y d = p3a::determinant(td.V);

    EXPECT_FLOAT_EQ(-0.6860431470481314,d) << "Determinant of p3a::mandel6x1 V";
}

TEST(mandel_tensors,LinAlg2ndOrderCxV){

    TestData td;
    p3a::mandel6x1 T = td.C * td.V;

    EXPECT_FLOAT_EQ(3.4635476194506110,T.x1()) << "T.x1";
    EXPECT_FLOAT_EQ(2.8336057738272911,T.x2()) << "T.x2";
    EXPECT_FLOAT_EQ(2.3772421613021302,T.x3()) << "T.x3";
    EXPECT_FLOAT_EQ(2.3851731903163613,T.x4()) << "T.x4";
    EXPECT_FLOAT_EQ(3.9147136807559804,T.x5()) << "T.x5";
    EXPECT_FLOAT_EQ(3.0036995840611000,T.x6()) << "T.x6";
}

TEST(mandel_tensors,LinAlg2ndOrderCxTensorV){

    TestData td;
    p3a::mandel6x1 T = td.C*td.TV;

    EXPECT_FLOAT_EQ(3.4635476194506110,T.x1()) << "T.x1";
    EXPECT_FLOAT_EQ(2.8336057738272911,T.x2()) << "T.x2";
    EXPECT_FLOAT_EQ(2.3772421613021302,T.x3()) << "T.x3";
    EXPECT_FLOAT_EQ(2.3851731903163613,T.x4()) << "T.x4";
    EXPECT_FLOAT_EQ(3.9147136807559804,T.x5()) << "T.x5";
    EXPECT_FLOAT_EQ(3.0036995840611000,T.x6()) << "T.x6";
}

TEST(mandel_tensors,LinAlg2ndOrdere63xVectorv){

    TestData td;
    p3a::mandel6x1 T = td.e63*td.v;

    EXPECT_FLOAT_EQ(0.7543830564053962,T.x1()) << "T.x1";
    EXPECT_FLOAT_EQ(0.8455346292231598,T.x2()) << "T.x2";
    EXPECT_FLOAT_EQ(0.4916193894817055,T.x3()) << "T.x3";
    EXPECT_FLOAT_EQ(0.5169985553863012,T.x4()) << "T.x4";
    EXPECT_FLOAT_EQ(0.8237161638682240,T.x5()) << "T.x5";
    EXPECT_FLOAT_EQ(0.3581252106713043,T.x6()) << "T.x6";
}

/**************************************************************************
 * physlib::vector3 Conversion tests
 *************************************************************************/
TEST(mandel_tensors,VectorConvTestse36xV){

    TestData td;
    p3a::vector3 T = td.e36*td.V;

    EXPECT_FLOAT_EQ(1.1959255529807966,T.x()) << "T.x";
    EXPECT_FLOAT_EQ(0.9246490053867547,T.y()) << "T.y";
    EXPECT_FLOAT_EQ(2.0158676300370968,T.z()) << "T.z";
}

TEST(mandel_tensors,VectorConvTestsTensorxVector3){

    TestData td;
    p3a::vector3 T = td.TV*td.v;

    EXPECT_FLOAT_EQ(0.3466382442875668,T.x()) << "T.x";
    EXPECT_FLOAT_EQ(1.2363564442415020,T.y()) << "T.y";
    EXPECT_FLOAT_EQ(0.6637312240540059,T.z()) << "T.z";
}

/**************************************************************************
 * Tensor Conversion tests
 *************************************************************************/
TEST(mandel_tensors,TensorConvTestsMandelToTensor){

    TestData td;
    const auto & VinvX = td.VinvX;
    p3a::matrix3x3 T = p3a::mandel6x1_to_matrix3x3(td.V);

    EXPECT_FLOAT_EQ(VinvX.x1(),T.xx()) << "T.xx()";
    EXPECT_FLOAT_EQ(VinvX.x6(),T.xy()) << "T.xy()";
    EXPECT_FLOAT_EQ(VinvX.x5(),T.xz()) << "T.xz()";
    EXPECT_FLOAT_EQ(VinvX.x6(),T.yx()) << "T.yx()";
    EXPECT_FLOAT_EQ(VinvX.x2(),T.yy()) << "T.yy()";
    EXPECT_FLOAT_EQ(VinvX.x4(),T.yz()) << "T.yz()";
    EXPECT_FLOAT_EQ(VinvX.x5(),T.zx()) << "T.zx()";
    EXPECT_FLOAT_EQ(VinvX.x4(),T.zy()) << "T.zy()";
    EXPECT_FLOAT_EQ(VinvX.x3(),T.zz()) << "T.zz()";
}

TEST(mandel_tensors,TensorConvTestse36xe63){

    TestData td;
    p3a::matrix3x3 T = td.e36*td.e63;

    EXPECT_FLOAT_EQ(0.8131190728640454,T.xx()) << "T.xx()";
    EXPECT_FLOAT_EQ(0.9959957879323522,T.xy()) << "T.xy()";
    EXPECT_FLOAT_EQ(0.5743955040157696,T.xz()) << "T.xz()";
    EXPECT_FLOAT_EQ(0.7556684864696379,T.yx()) << "T.yx()";
    EXPECT_FLOAT_EQ(0.6033617978655575,T.yy()) << "T.yy()";
    EXPECT_FLOAT_EQ(0.7408493369039156,T.yz()) << "T.yz()";
    EXPECT_FLOAT_EQ(1.5297772340856588,T.zx()) << "T.zx()";
    EXPECT_FLOAT_EQ(1.2950737798283347,T.zy()) << "T.zy()";
    EXPECT_FLOAT_EQ(1.2531807400797446,T.zz()) << "T.zz()";
}

TEST(mandel_tensors,TensorConvTestsVxTensor){

    TestData td;
    p3a::matrix3x3 T = td.V*td.TV;

    EXPECT_FLOAT_EQ(0.7869648629834085,T.xx()) << "T.xx()";
    EXPECT_FLOAT_EQ(0.4139354501342279,T.xy()) << "T.xy()";
    EXPECT_FLOAT_EQ(0.7935811988267519,T.xz()) << "T.xz()";
    EXPECT_FLOAT_EQ(0.4139354501342279,T.yx()) << "T.yx()";
    EXPECT_FLOAT_EQ(1.6263493443669228,T.yy()) << "T.yy()";
    EXPECT_FLOAT_EQ(0.7360230649814389,T.yz()) << "T.yz()";
    EXPECT_FLOAT_EQ(0.7935811988267519,T.zx()) << "T.zx()";
    EXPECT_FLOAT_EQ(0.7360230649814389,T.zy()) << "T.zy()";
    EXPECT_FLOAT_EQ(1.2968901375823718,T.zz()) << "T.zz()";
}

TEST(mandel_tensors,TensorConvTestsTensorxV){

    TestData td;
    p3a::matrix3x3 T = td.TV*td.V;

    EXPECT_FLOAT_EQ(0.7869648629834085,T.xx()) << "T.xx()";
    EXPECT_FLOAT_EQ(0.4139354501342279,T.xy()) << "T.xy()";
    EXPECT_FLOAT_EQ(0.7935811988267519,T.xz()) << "T.xz()";
    EXPECT_FLOAT_EQ(0.4139354501342279,T.yx()) << "T.yx()";
    EXPECT_FLOAT_EQ(1.6263493443669228,T.yy()) << "T.yy()";
    EXPECT_FLOAT_EQ(0.7360230649814389,T.yz()) << "T.yz()";
    EXPECT_FLOAT_EQ(0.7935811988267519,T.zx()) << "T.zx()";
    EXPECT_FLOAT_EQ(0.7360230649814389,T.zy()) << "T.zy()";
    EXPECT_FLOAT_EQ(1.2968901375823718,T.zz()) << "T.zz()";
}

TEST(mandel_tensors,TensorConvTestsTensorxTensor){

    TestData td;
    p3a::matrix3x3 T = td.TV*td.TV;

    EXPECT_FLOAT_EQ(0.7869648629834085,T.xx()) << "T.xx()";
    EXPECT_FLOAT_EQ(0.4139354501342279,T.xy()) << "T.xy()";
    EXPECT_FLOAT_EQ(0.7935811988267519,T.xz()) << "T.xz()";
    EXPECT_FLOAT_EQ(0.4139354501342279,T.yx()) << "T.yx()";
    EXPECT_FLOAT_EQ(1.6263493443669228,T.yy()) << "T.yy()";
    EXPECT_FLOAT_EQ(0.7360230649814389,T.yz()) << "T.yz()";
    EXPECT_FLOAT_EQ(0.7935811988267519,T.zx()) << "T.zx()";
    EXPECT_FLOAT_EQ(0.7360230649814389,T.zy()) << "T.zy()";
    EXPECT_FLOAT_EQ(1.2968901375823718,T.zz()) << "T.zz()";
}

/**************************************************************************
 * symmetric3x3 Conversion tests
 *************************************************************************/
TEST(mandel_tensors,symmetric3x3ConvTestsMandelVectorConstructor){

    TestData td;
    const auto & V = td.V;
    const auto & VinvX = td.VinvX;
    p3a::symmetric3x3 U = p3a::mandel6x1_to_symmetric3x3(V);

    EXPECT_FLOAT_EQ(VinvX.x1(),U.xx()) << "U.x1";
    EXPECT_FLOAT_EQ(VinvX.x2(),U.yy()) << "U.x2";
    EXPECT_FLOAT_EQ(VinvX.x3(),U.zz()) << "U.x3";
    EXPECT_FLOAT_EQ(VinvX.x4(),U.yz()) << "U.x4";
    EXPECT_FLOAT_EQ(VinvX.x5(),U.xz()) << "U.x5";
    EXPECT_FLOAT_EQ(VinvX.x6(),U.xy()) << "U.x6";
}

/**************************************************************************
 * Constructor tests for 4th-order MandelTensors:
 *************************************************************************/
TEST(mandel_tensors,Construct4thOrderCfromListxform){

    //already done, made from initial setup  
    TestData td;
    const auto &C = td.C;

    EXPECT_FLOAT_EQ(0.4335153608215544,C.x11()) << "C.x11()";
    EXPECT_FLOAT_EQ(0.0856007508096491,C.x12()) << "C.x12()";
    EXPECT_FLOAT_EQ(0.6181864586228960,C.x13()) << "C.x13()";
    EXPECT_FLOAT_EQ(1.0180326366236518,C.x14()) << "C.x14()";
    EXPECT_FLOAT_EQ(0.9998149495390550,C.x15()) << "C.x15()";
    EXPECT_FLOAT_EQ(1.3248978263217353,C.x16()) << "C.x16()";
    EXPECT_FLOAT_EQ(0.4327952022077987,C.x21()) << "C.x21()";
    EXPECT_FLOAT_EQ(0.3742135173559312,C.x22()) << "C.x22()";
    EXPECT_FLOAT_EQ(0.1080020759711545,C.x23()) << "C.x23()";
    EXPECT_FLOAT_EQ(0.8147725263605445,C.x24()) << "C.x24()";
    EXPECT_FLOAT_EQ(0.9048931279343728,C.x25()) << "C.x25()";
    EXPECT_FLOAT_EQ(1.2476465544083184,C.x26()) << "C.x26()";
    EXPECT_FLOAT_EQ(0.1619267043337781,C.x31()) << "C.x31()";
    EXPECT_FLOAT_EQ(0.8255468602509465,C.x32()) << "C.x32()";
    EXPECT_FLOAT_EQ(0.4823784238849611,C.x33()) << "C.x33()";
    EXPECT_FLOAT_EQ(0.6837503075397613,C.x34()) << "C.x34()";
    EXPECT_FLOAT_EQ(0.0784973317433504,C.x35()) << "C.x35()";
    EXPECT_FLOAT_EQ(0.8368479998334433,C.x36()) << "C.x36()";
    EXPECT_FLOAT_EQ(0.0112200396694362,C.x41()) << "C.x41()";
    EXPECT_FLOAT_EQ(0.8194354458300377,C.x42()) << "C.x42()";
    EXPECT_FLOAT_EQ(1.1074040005476036,C.x43()) << "C.x43()";
    EXPECT_FLOAT_EQ(0.6118023983888154,C.x44()) << "C.x44()";
    EXPECT_FLOAT_EQ(0.4369542629777867,C.x45()) << "C.x45()";
    EXPECT_FLOAT_EQ(0.6474343145633832,C.x46()) << "C.x46()";
    EXPECT_FLOAT_EQ(0.0836906759361557,C.x51()) << "C.x51()";
    EXPECT_FLOAT_EQ(0.9338502039452401,C.x52()) << "C.x52()";
    EXPECT_FLOAT_EQ(0.9911298229891556,C.x53()) << "C.x53()";
    EXPECT_FLOAT_EQ(1.6059479559586896,C.x54()) << "C.x54()";
    EXPECT_FLOAT_EQ(1.9166663623233735,C.x55()) << "C.x55()";
    EXPECT_FLOAT_EQ(0.8109702524757447,C.x56()) << "C.x56()";
    EXPECT_FLOAT_EQ(0.0501618784358511,C.x61()) << "C.x61()";
    EXPECT_FLOAT_EQ(0.9139519661627958,C.x62()) << "C.x62()";
    EXPECT_FLOAT_EQ(0.4597365381087316,C.x63()) << "C.x63()";
    EXPECT_FLOAT_EQ(1.0569116390136390,C.x64()) << "C.x64()";
    EXPECT_FLOAT_EQ(0.6655431451182421,C.x65()) << "C.x65()";
    EXPECT_FLOAT_EQ(0.9678972164817718,C.x66()) << "C.x66()";
}

TEST(mandel_tensors,Construct4thOrderCfromC){

    //already done, made from initial setup  
    TestData td;
    const auto &C(td.C);

    EXPECT_FLOAT_EQ(0.4335153608215544,C.x11()) << "C.x11()";
    EXPECT_FLOAT_EQ(0.0856007508096491,C.x12()) << "C.x12()";
    EXPECT_FLOAT_EQ(0.6181864586228960,C.x13()) << "C.x13()";
    EXPECT_FLOAT_EQ(1.0180326366236518,C.x14()) << "C.x14()";
    EXPECT_FLOAT_EQ(0.9998149495390550,C.x15()) << "C.x15()";
    EXPECT_FLOAT_EQ(1.3248978263217353,C.x16()) << "C.x16()";
    EXPECT_FLOAT_EQ(0.4327952022077987,C.x21()) << "C.x21()";
    EXPECT_FLOAT_EQ(0.3742135173559312,C.x22()) << "C.x22()";
    EXPECT_FLOAT_EQ(0.1080020759711545,C.x23()) << "C.x23()";
    EXPECT_FLOAT_EQ(0.8147725263605445,C.x24()) << "C.x24()";
    EXPECT_FLOAT_EQ(0.9048931279343728,C.x25()) << "C.x25()";
    EXPECT_FLOAT_EQ(1.2476465544083184,C.x26()) << "C.x26()";
    EXPECT_FLOAT_EQ(0.1619267043337781,C.x31()) << "C.x31()";
    EXPECT_FLOAT_EQ(0.8255468602509465,C.x32()) << "C.x32()";
    EXPECT_FLOAT_EQ(0.4823784238849611,C.x33()) << "C.x33()";
    EXPECT_FLOAT_EQ(0.6837503075397613,C.x34()) << "C.x34()";
    EXPECT_FLOAT_EQ(0.0784973317433504,C.x35()) << "C.x35()";
    EXPECT_FLOAT_EQ(0.8368479998334433,C.x36()) << "C.x36()";
    EXPECT_FLOAT_EQ(0.0112200396694362,C.x41()) << "C.x41()";
    EXPECT_FLOAT_EQ(0.8194354458300377,C.x42()) << "C.x42()";
    EXPECT_FLOAT_EQ(1.1074040005476036,C.x43()) << "C.x43()";
    EXPECT_FLOAT_EQ(0.6118023983888154,C.x44()) << "C.x44()";
    EXPECT_FLOAT_EQ(0.4369542629777867,C.x45()) << "C.x45()";
    EXPECT_FLOAT_EQ(0.6474343145633832,C.x46()) << "C.x46()";
    EXPECT_FLOAT_EQ(0.0836906759361557,C.x51()) << "C.x51()";
    EXPECT_FLOAT_EQ(0.9338502039452401,C.x52()) << "C.x52()";
    EXPECT_FLOAT_EQ(0.9911298229891556,C.x53()) << "C.x53()";
    EXPECT_FLOAT_EQ(1.6059479559586896,C.x54()) << "C.x54()";
    EXPECT_FLOAT_EQ(1.9166663623233735,C.x55()) << "C.x55()";
    EXPECT_FLOAT_EQ(0.8109702524757447,C.x56()) << "C.x56()";
    EXPECT_FLOAT_EQ(0.0501618784358511,C.x61()) << "C.x61()";
    EXPECT_FLOAT_EQ(0.9139519661627958,C.x62()) << "C.x62()";
    EXPECT_FLOAT_EQ(0.4597365381087316,C.x63()) << "C.x63()";
    EXPECT_FLOAT_EQ(1.0569116390136390,C.x64()) << "C.x64()";
    EXPECT_FLOAT_EQ(0.6655431451182421,C.x65()) << "C.x65()";
    EXPECT_FLOAT_EQ(0.9678972164817718,C.x66()) << "C.x66()";
}

TEST(mandel_tensors,Construct4thOrderCfromstaticmatrixxform){

    TestData td;
    const p3a::mandel6x6<Y> C(td.Cstatic);

    EXPECT_FLOAT_EQ(0.4335153608215544,C.x11()) << "C.x11()";
    EXPECT_FLOAT_EQ(0.0856007508096491,C.x12()) << "C.x12()";
    EXPECT_FLOAT_EQ(0.6181864586228960,C.x13()) << "C.x13()";
    EXPECT_FLOAT_EQ(1.0180326366236518,C.x14()) << "C.x14()";
    EXPECT_FLOAT_EQ(0.9998149495390550,C.x15()) << "C.x15()";
    EXPECT_FLOAT_EQ(1.3248978263217353,C.x16()) << "C.x16()";
    EXPECT_FLOAT_EQ(0.4327952022077987,C.x21()) << "C.x21()";
    EXPECT_FLOAT_EQ(0.3742135173559312,C.x22()) << "C.x22()";
    EXPECT_FLOAT_EQ(0.1080020759711545,C.x23()) << "C.x23()";
    EXPECT_FLOAT_EQ(0.8147725263605445,C.x24()) << "C.x24()";
    EXPECT_FLOAT_EQ(0.9048931279343728,C.x25()) << "C.x25()";
    EXPECT_FLOAT_EQ(1.2476465544083184,C.x26()) << "C.x26()";
    EXPECT_FLOAT_EQ(0.1619267043337781,C.x31()) << "C.x31()";
    EXPECT_FLOAT_EQ(0.8255468602509465,C.x32()) << "C.x32()";
    EXPECT_FLOAT_EQ(0.4823784238849611,C.x33()) << "C.x33()";
    EXPECT_FLOAT_EQ(0.6837503075397613,C.x34()) << "C.x34()";
    EXPECT_FLOAT_EQ(0.0784973317433504,C.x35()) << "C.x35()";
    EXPECT_FLOAT_EQ(0.8368479998334433,C.x36()) << "C.x36()";
    EXPECT_FLOAT_EQ(0.0112200396694362,C.x41()) << "C.x41()";
    EXPECT_FLOAT_EQ(0.8194354458300377,C.x42()) << "C.x42()";
    EXPECT_FLOAT_EQ(1.1074040005476036,C.x43()) << "C.x43()";
    EXPECT_FLOAT_EQ(0.6118023983888154,C.x44()) << "C.x44()";
    EXPECT_FLOAT_EQ(0.4369542629777867,C.x45()) << "C.x45()";
    EXPECT_FLOAT_EQ(0.6474343145633832,C.x46()) << "C.x46()";
    EXPECT_FLOAT_EQ(0.0836906759361557,C.x51()) << "C.x51()";
    EXPECT_FLOAT_EQ(0.9338502039452401,C.x52()) << "C.x52()";
    EXPECT_FLOAT_EQ(0.9911298229891556,C.x53()) << "C.x53()";
    EXPECT_FLOAT_EQ(1.6059479559586896,C.x54()) << "C.x54()";
    EXPECT_FLOAT_EQ(1.9166663623233735,C.x55()) << "C.x55()";
    EXPECT_FLOAT_EQ(0.8109702524757447,C.x56()) << "C.x56()";
    EXPECT_FLOAT_EQ(0.0501618784358511,C.x61()) << "C.x61()";
    EXPECT_FLOAT_EQ(0.9139519661627958,C.x62()) << "C.x62()";
    EXPECT_FLOAT_EQ(0.4597365381087316,C.x63()) << "C.x63()";
    EXPECT_FLOAT_EQ(1.0569116390136390,C.x64()) << "C.x64()";
    EXPECT_FLOAT_EQ(0.6655431451182421,C.x65()) << "C.x65()";
    EXPECT_FLOAT_EQ(0.9678972164817718,C.x66()) << "C.x66()";
}

/**************************************************************************
 * Basic Operations 4th-order MandelTensors
 *************************************************************************/
TEST(mandel_tensors,Basics4thOrderCpluseqCp){

    TestData td;
    const auto & C = td.C;
    const auto & Cp = td.Cp;
    p3a::mandel6x6 W = C;
    W += Cp;

    EXPECT_FLOAT_EQ(C.x11()+Cp.x11(),W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()+Cp.x12(),W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()+Cp.x13(),W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()+Cp.x14(),W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()+Cp.x15(),W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()+Cp.x16(),W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()+Cp.x21(),W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()+Cp.x22(),W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()+Cp.x23(),W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()+Cp.x24(),W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()+Cp.x25(),W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()+Cp.x26(),W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()+Cp.x31(),W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()+Cp.x32(),W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()+Cp.x33(),W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()+Cp.x34(),W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()+Cp.x35(),W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()+Cp.x36(),W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()+Cp.x41(),W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()+Cp.x42(),W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()+Cp.x43(),W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()+Cp.x44(),W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()+Cp.x45(),W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()+Cp.x46(),W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()+Cp.x51(),W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()+Cp.x52(),W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()+Cp.x53(),W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()+Cp.x54(),W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()+Cp.x55(),W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()+Cp.x56(),W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()+Cp.x61(),W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()+Cp.x62(),W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()+Cp.x63(),W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()+Cp.x64(),W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()+Cp.x65(),W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()+Cp.x66(),W.x66()) << "T.x66()";
}

TEST(mandel_tensors,Basics4thOrderCminuseqCp){

    TestData td;
    const auto & C = td.C;
    const auto & Cp = td.Cp;
    p3a::mandel6x6 W = C;
    W -= Cp;
    EXPECT_FLOAT_EQ(C.x11()-Cp.x11(),W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()-Cp.x12(),W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()-Cp.x13(),W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()-Cp.x14(),W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()-Cp.x15(),W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()-Cp.x16(),W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()-Cp.x21(),W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()-Cp.x22(),W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()-Cp.x23(),W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()-Cp.x24(),W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()-Cp.x25(),W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()-Cp.x26(),W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()-Cp.x31(),W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()-Cp.x32(),W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()-Cp.x33(),W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()-Cp.x34(),W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()-Cp.x35(),W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()-Cp.x36(),W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()-Cp.x41(),W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()-Cp.x42(),W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()-Cp.x43(),W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()-Cp.x44(),W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()-Cp.x45(),W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()-Cp.x46(),W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()-Cp.x51(),W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()-Cp.x52(),W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()-Cp.x53(),W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()-Cp.x54(),W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()-Cp.x55(),W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()-Cp.x56(),W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()-Cp.x61(),W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()-Cp.x62(),W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()-Cp.x63(),W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()-Cp.x64(),W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()-Cp.x65(),W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()-Cp.x66(),W.x66()) << "T.x66()";
}

TEST(mandel_tensors,Basics4thOrderCaddCp){

    TestData td;
    const auto & C = td.C;
    const auto & Cp = td.Cp;
    p3a::mandel6x6 W = Cp + C;

    EXPECT_FLOAT_EQ(C.x11()+Cp.x11(),W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()+Cp.x12(),W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()+Cp.x13(),W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()+Cp.x14(),W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()+Cp.x15(),W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()+Cp.x16(),W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()+Cp.x21(),W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()+Cp.x22(),W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()+Cp.x23(),W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()+Cp.x24(),W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()+Cp.x25(),W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()+Cp.x26(),W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()+Cp.x31(),W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()+Cp.x32(),W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()+Cp.x33(),W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()+Cp.x34(),W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()+Cp.x35(),W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()+Cp.x36(),W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()+Cp.x41(),W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()+Cp.x42(),W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()+Cp.x43(),W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()+Cp.x44(),W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()+Cp.x45(),W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()+Cp.x46(),W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()+Cp.x51(),W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()+Cp.x52(),W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()+Cp.x53(),W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()+Cp.x54(),W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()+Cp.x55(),W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()+Cp.x56(),W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()+Cp.x61(),W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()+Cp.x62(),W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()+Cp.x63(),W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()+Cp.x64(),W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()+Cp.x65(),W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()+Cp.x66(),W.x66()) << "T.x66()";
}

TEST(mandel_tensors,Basics4thOrderCminusCp){

    TestData td;
    const auto & C = td.C;
    const auto & Cp = td.Cp;
    p3a::mandel6x6 W = C - Cp;

    EXPECT_FLOAT_EQ(C.x11()-Cp.x11(),W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()-Cp.x12(),W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()-Cp.x13(),W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()-Cp.x14(),W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()-Cp.x15(),W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()-Cp.x16(),W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()-Cp.x21(),W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()-Cp.x22(),W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()-Cp.x23(),W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()-Cp.x24(),W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()-Cp.x25(),W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()-Cp.x26(),W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()-Cp.x31(),W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()-Cp.x32(),W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()-Cp.x33(),W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()-Cp.x34(),W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()-Cp.x35(),W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()-Cp.x36(),W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()-Cp.x41(),W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()-Cp.x42(),W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()-Cp.x43(),W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()-Cp.x44(),W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()-Cp.x45(),W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()-Cp.x46(),W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()-Cp.x51(),W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()-Cp.x52(),W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()-Cp.x53(),W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()-Cp.x54(),W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()-Cp.x55(),W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()-Cp.x56(),W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()-Cp.x61(),W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()-Cp.x62(),W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()-Cp.x63(),W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()-Cp.x64(),W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()-Cp.x65(),W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()-Cp.x66(),W.x66()) << "T.x66()";
}                       

TEST(mandel_tensors,Basics4thOrderCxReal){

    TestData td;
    const auto & C = td.C;
    const auto & r1 = td.r1;
    p3a::mandel6x6 W = C;
    W*=r1;

    EXPECT_FLOAT_EQ(C.x11()*r1,W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()*r1,W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()*r1,W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()*r1,W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()*r1,W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()*r1,W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()*r1,W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()*r1,W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()*r1,W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()*r1,W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()*r1,W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()*r1,W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()*r1,W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()*r1,W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()*r1,W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()*r1,W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()*r1,W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()*r1,W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()*r1,W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()*r1,W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()*r1,W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()*r1,W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()*r1,W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()*r1,W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()*r1,W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()*r1,W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()*r1,W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()*r1,W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()*r1,W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()*r1,W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()*r1,W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()*r1,W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()*r1,W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()*r1,W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()*r1,W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()*r1,W.x66()) << "T.x66()";
}

TEST(mandel_tensors,Basics4thOrderBinaryCxReal){

    TestData td;
    const auto & C = td.C;
    const auto & r1 = td.r1;
    p3a::mandel6x6 W = C*r1;

    EXPECT_FLOAT_EQ(C.x11()*r1,W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()*r1,W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()*r1,W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()*r1,W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()*r1,W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()*r1,W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()*r1,W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()*r1,W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()*r1,W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()*r1,W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()*r1,W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()*r1,W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()*r1,W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()*r1,W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()*r1,W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()*r1,W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()*r1,W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()*r1,W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()*r1,W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()*r1,W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()*r1,W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()*r1,W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()*r1,W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()*r1,W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()*r1,W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()*r1,W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()*r1,W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()*r1,W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()*r1,W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()*r1,W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()*r1,W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()*r1,W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()*r1,W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()*r1,W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()*r1,W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()*r1,W.x66()) << "T.x66()";
}

TEST(mandel_tensors,Basics4thOrderBinaryRealxC){

    TestData td;
    const auto & C = td.C;
    const auto & r1 = td.r1;
    p3a::mandel6x6 W = r1 * C;

    EXPECT_FLOAT_EQ(C.x11()*r1,W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()*r1,W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()*r1,W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()*r1,W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()*r1,W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()*r1,W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()*r1,W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()*r1,W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()*r1,W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()*r1,W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()*r1,W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()*r1,W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()*r1,W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()*r1,W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()*r1,W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()*r1,W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()*r1,W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()*r1,W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()*r1,W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()*r1,W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()*r1,W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()*r1,W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()*r1,W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()*r1,W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()*r1,W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()*r1,W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()*r1,W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()*r1,W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()*r1,W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()*r1,W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()*r1,W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()*r1,W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()*r1,W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()*r1,W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()*r1,W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()*r1,W.x66()) << "T.x66()";
}

TEST(mandel_tensors,Basics4thOrderCdivReal){

    TestData td;
    const auto & C = td.C;
    const auto & r1 = td.r1;
    p3a::mandel6x6 W = C;
    W/=r1;

    EXPECT_FLOAT_EQ(C.x11()/r1,W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()/r1,W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()/r1,W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()/r1,W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()/r1,W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()/r1,W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()/r1,W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()/r1,W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()/r1,W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()/r1,W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()/r1,W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()/r1,W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()/r1,W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()/r1,W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()/r1,W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()/r1,W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()/r1,W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()/r1,W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()/r1,W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()/r1,W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()/r1,W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()/r1,W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()/r1,W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()/r1,W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()/r1,W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()/r1,W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()/r1,W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()/r1,W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()/r1,W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()/r1,W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()/r1,W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()/r1,W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()/r1,W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()/r1,W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()/r1,W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()/r1,W.x66()) << "T.x66()";
}

TEST(mandel_tensors,Basics4thOrderBinaryCdivReal){

    TestData td;
    const auto & C = td.C;
    const auto & r1 = td.r1;
    p3a::mandel6x6 W = C/r1;

    EXPECT_FLOAT_EQ(C.x11()/r1,W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x12()/r1,W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x13()/r1,W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x14()/r1,W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x15()/r1,W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x16()/r1,W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x21()/r1,W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22()/r1,W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x23()/r1,W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x24()/r1,W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x25()/r1,W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x26()/r1,W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x31()/r1,W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x32()/r1,W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33()/r1,W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x34()/r1,W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x35()/r1,W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x36()/r1,W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x41()/r1,W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x42()/r1,W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x43()/r1,W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44()/r1,W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x45()/r1,W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x46()/r1,W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x51()/r1,W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x52()/r1,W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x53()/r1,W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x54()/r1,W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55()/r1,W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x56()/r1,W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x61()/r1,W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x62()/r1,W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x63()/r1,W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x64()/r1,W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x65()/r1,W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66()/r1,W.x66()) << "T.x66()";
}

/**************************************************************************
 * Member function tests for 4th-order MandelTensors
 *************************************************************************/
TEST(mandel_tensors,LinAlg4thOrderMembersTransposeC){

    TestData td;
    const auto & C = td.C;
    p3a::mandel6x6 W = transpose(C);

    EXPECT_FLOAT_EQ(C.x11(),W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(C.x21(),W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(C.x31(),W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(C.x41(),W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(C.x51(),W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(C.x61(),W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(C.x12(),W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(C.x22(),W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(C.x32(),W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(C.x42(),W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(C.x52(),W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(C.x62(),W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(C.x13(),W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(C.x23(),W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(C.x33(),W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(C.x43(),W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(C.x53(),W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(C.x63(),W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(C.x14(),W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(C.x24(),W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(C.x34(),W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(C.x44(),W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(C.x54(),W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(C.x64(),W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(C.x15(),W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(C.x25(),W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(C.x35(),W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(C.x45(),W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(C.x55(),W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(C.x65(),W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(C.x16(),W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(C.x26(),W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(C.x36(),W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(C.x46(),W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(C.x56(),W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(C.x66(),W.x66()) << "T.x66()";
}

TEST(mandel_tensors,LinAlg4thOrderCxCp){

    TestData td;
    p3a::mandel6x6 W = td.C*td.Cp;

    EXPECT_FLOAT_EQ(3.4324641635231457,W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(3.1652841587579554,W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(3.7807607038941047,W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(4.2929990275344521,W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(5.6740312379032591,W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(6.6696446443745385,W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(2.8842299333474313,W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(2.5497370745168593,W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(3.1998897121280914,W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(3.4733946969525569,W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(4.9716389699058672,W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(5.8233675439417523,W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(2.6320661860595416,W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(2.2593857295548627,W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(1.9106598610937264,W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(2.7578998057721957,W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(3.6603206355467375,W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(4.2128432475707616,W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(3.2441769156729801,W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(2.5142287416369151,W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(2.2091631392392084,W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(3.8997518624635563,W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(4.7473966292929779,W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(4.9565477090185430,W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(5.5339581850477266,W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(3.6804177173549402,W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(4.2860776978446804,W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(8.0210977617023076,W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(8.3997855773870853,W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(8.9870653250055597,W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(3.5914517574927998,W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(2.7927266821323165,W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(2.7422796353117542,W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(4.4261334775767400,W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(5.2442699849545713,W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(5.8135596538845116,W.x66()) << "T.x66()";
}

TEST(mandel_tensors,LinAlg4thOrderdote63xe36){

    TestData td;
    p3a::mandel6x6 W = td.e63*td.e36;

    EXPECT_FLOAT_EQ(0.6481000136992765,W.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(0.4371968248688184,W.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(0.4567401931759338,W.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(0.5084467410380618,W.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(0.8658538452444362,W.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(0.8270178894938859,W.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(0.9406952522707415,W.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(0.2483595248852332,W.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(0.4811208705993483,W.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(0.4424702363903814,W.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(0.4670224166308748,W.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(0.6944279990001526,W.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(0.6348517948283502,W.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(0.1063141324215460,W.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(0.2969290536912734,W.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(0.2262760176533510,W.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(0.2549385207966328,W.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(0.4362317224266326,W.x36()) << "T.x36()";
    EXPECT_FLOAT_EQ(0.5209422981631543,W.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(0.1946090556325360,W.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(0.2929761942755778,W.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(0.2963774815244760,W.x44()) << "T.x44()";
    EXPECT_FLOAT_EQ(0.3556848018688814,W.x45()) << "T.x45()";
    EXPECT_FLOAT_EQ(0.4448801241685245,W.x46()) << "T.x46()";
    EXPECT_FLOAT_EQ(0.6706444943197795,W.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(0.4484542729222799,W.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(0.4695062540258030,W.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(0.5525554002871033,W.x54()) << "T.x54()";
    EXPECT_FLOAT_EQ(0.8125274881895002,W.x55()) << "T.x55()";
    EXPECT_FLOAT_EQ(0.7951750536126074,W.x56()) << "T.x56()";
    EXPECT_FLOAT_EQ(0.3032681701367076,W.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(0.1981818384668978,W.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(0.2104566600814864,W.x63()) << "T.x63()";
    EXPECT_FLOAT_EQ(0.2389951119718454,W.x64()) << "T.x64()";
    EXPECT_FLOAT_EQ(0.3767984844926149,W.x65()) << "T.x65()";
    EXPECT_FLOAT_EQ(0.3673680488195881,W.x66()) << "T.x66()";
}

/**************************************************************************
 * Constructor tests for 3rd order tensors (MandelTensor36)
 *************************************************************************/
TEST(mandel_tensors,Construct36e36fromListxform){

    TestData td;
    const auto & e36 = td.e36;

    //already completed in initlization
    EXPECT_FLOAT_EQ(0.8076127837790422,e36.x11()) << "e36.x11()";
    EXPECT_FLOAT_EQ(0.0119054861420681,e36.x12()) << "e36.x12()";
    EXPECT_FLOAT_EQ(0.3198164644230427,e36.x13()) << "e36.x13()";
    EXPECT_FLOAT_EQ(0.1899714025547371,e36.x14()) << "e36.x14()";
    EXPECT_FLOAT_EQ(0.0810795183410036,e36.x15()) << "e36.x15()";
    EXPECT_FLOAT_EQ(0.3999599660633990,e36.x16()) << "e36.x16()";
    EXPECT_FLOAT_EQ(0.0453167988352610,e36.x21()) << "e36.x21()";
    EXPECT_FLOAT_EQ(0.2818248347311008,e36.x22()) << "e36.x22()";
    EXPECT_FLOAT_EQ(0.1508671814551668,e36.x23()) << "e36.x23()";
    EXPECT_FLOAT_EQ(0.2127405166893033,e36.x24()) << "e36.x24()";
    EXPECT_FLOAT_EQ(0.6128683635764686,e36.x25()) << "e36.x25()";
    EXPECT_FLOAT_EQ(0.4153444264017785,e36.x26()) << "e36.x26()";
    EXPECT_FLOAT_EQ(0.5405046604793561,e36.x31()) << "e36.x31()";
    EXPECT_FLOAT_EQ(0.4391783723249735,e36.x32()) << "e36.x32()";
    EXPECT_FLOAT_EQ(0.4128631387354691,e36.x33()) << "e36.x33()";
    EXPECT_FLOAT_EQ(0.5546654249524924,e36.x34()) << "e36.x34()";
    EXPECT_FLOAT_EQ(0.6862523687819362,e36.x35()) << "e36.x35()";
    EXPECT_FLOAT_EQ(0.6488819206134113,e36.x36()) << "e36.x36()";
}

TEST(mandel_tensors,Construct36e36frome36){

    TestData td;
    const auto & e36(td.e36);

    //already completed in initlization
    EXPECT_FLOAT_EQ(0.8076127837790422,e36.x11()) << "e36.x11()";
    EXPECT_FLOAT_EQ(0.0119054861420681,e36.x12()) << "e36.x12()";
    EXPECT_FLOAT_EQ(0.3198164644230427,e36.x13()) << "e36.x13()";
    EXPECT_FLOAT_EQ(0.1899714025547371,e36.x14()) << "e36.x14()";
    EXPECT_FLOAT_EQ(0.0810795183410036,e36.x15()) << "e36.x15()";
    EXPECT_FLOAT_EQ(0.3999599660633990,e36.x16()) << "e36.x16()";
    EXPECT_FLOAT_EQ(0.0453167988352610,e36.x21()) << "e36.x21()";
    EXPECT_FLOAT_EQ(0.2818248347311008,e36.x22()) << "e36.x22()";
    EXPECT_FLOAT_EQ(0.1508671814551668,e36.x23()) << "e36.x23()";
    EXPECT_FLOAT_EQ(0.2127405166893033,e36.x24()) << "e36.x24()";
    EXPECT_FLOAT_EQ(0.6128683635764686,e36.x25()) << "e36.x25()";
    EXPECT_FLOAT_EQ(0.4153444264017785,e36.x26()) << "e36.x26()";
    EXPECT_FLOAT_EQ(0.5405046604793561,e36.x31()) << "e36.x31()";
    EXPECT_FLOAT_EQ(0.4391783723249735,e36.x32()) << "e36.x32()";
    EXPECT_FLOAT_EQ(0.4128631387354691,e36.x33()) << "e36.x33()";
    EXPECT_FLOAT_EQ(0.5546654249524924,e36.x34()) << "e36.x34()";
    EXPECT_FLOAT_EQ(0.6862523687819362,e36.x35()) << "e36.x35()";
    EXPECT_FLOAT_EQ(0.6488819206134113,e36.x36()) << "e36.x36()";
}

TEST(mandel_tensors,Construct36e36frome36Xform){

    TestData td;
    const auto & e36invX = td.e36invX;
    p3a::mandel3x6 e(td.e36);
    e.invMandelXform();

    EXPECT_FLOAT_EQ(e36invX.x11(),e.x11()) << "e36.x11()";
    EXPECT_FLOAT_EQ(e36invX.x12(),e.x12()) << "e36.x12()";
    EXPECT_FLOAT_EQ(e36invX.x13(),e.x13()) << "e36.x13()";
    EXPECT_FLOAT_EQ(e36invX.x14(),e.x14()) << "e36.x14()";
    EXPECT_FLOAT_EQ(e36invX.x15(),e.x15()) << "e36.x15()";
    EXPECT_FLOAT_EQ(e36invX.x16(),e.x16()) << "e36.x16()";
    EXPECT_FLOAT_EQ(e36invX.x21(),e.x21()) << "e36.x21()";
    EXPECT_FLOAT_EQ(e36invX.x22(),e.x22()) << "e36.x22()";
    EXPECT_FLOAT_EQ(e36invX.x23(),e.x23()) << "e36.x23()";
    EXPECT_FLOAT_EQ(e36invX.x24(),e.x24()) << "e36.x24()";
    EXPECT_FLOAT_EQ(e36invX.x25(),e.x25()) << "e36.x25()";
    EXPECT_FLOAT_EQ(e36invX.x26(),e.x26()) << "e36.x26()";
    EXPECT_FLOAT_EQ(e36invX.x31(),e.x31()) << "e36.x31()";
    EXPECT_FLOAT_EQ(e36invX.x32(),e.x32()) << "e36.x32()";
    EXPECT_FLOAT_EQ(e36invX.x33(),e.x33()) << "e36.x33()";
    EXPECT_FLOAT_EQ(e36invX.x34(),e.x34()) << "e36.x34()";
    EXPECT_FLOAT_EQ(e36invX.x35(),e.x35()) << "e36.x35()";
    EXPECT_FLOAT_EQ(e36invX.x36(),e.x36()) << "e36.x36()";
}

TEST(mandel_tensors,Construct36e36fromList){

    TestData td;
    const auto & e36invX = td.e36invX;
    p3a::mandel3x6 e = td.e36;
    e.invMandelXform();

    EXPECT_FLOAT_EQ(e36invX.x11(),e.x11()) << "e36.x11()";
    EXPECT_FLOAT_EQ(e36invX.x12(),e.x12()) << "e36.x12()";
    EXPECT_FLOAT_EQ(e36invX.x13(),e.x13()) << "e36.x13()";
    EXPECT_FLOAT_EQ(e36invX.x14(),e.x14()) << "e36.x14()";
    EXPECT_FLOAT_EQ(e36invX.x15(),e.x15()) << "e36.x15()";
    EXPECT_FLOAT_EQ(e36invX.x16(),e.x16()) << "e36.x16()";
    EXPECT_FLOAT_EQ(e36invX.x21(),e.x21()) << "e36.x21()";
    EXPECT_FLOAT_EQ(e36invX.x22(),e.x22()) << "e36.x22()";
    EXPECT_FLOAT_EQ(e36invX.x23(),e.x23()) << "e36.x23()";
    EXPECT_FLOAT_EQ(e36invX.x24(),e.x24()) << "e36.x24()";
    EXPECT_FLOAT_EQ(e36invX.x25(),e.x25()) << "e36.x25()";
    EXPECT_FLOAT_EQ(e36invX.x26(),e.x26()) << "e36.x26()";
    EXPECT_FLOAT_EQ(e36invX.x31(),e.x31()) << "e36.x31()";
    EXPECT_FLOAT_EQ(e36invX.x32(),e.x32()) << "e36.x32()";
    EXPECT_FLOAT_EQ(e36invX.x33(),e.x33()) << "e36.x33()";
    EXPECT_FLOAT_EQ(e36invX.x34(),e.x34()) << "e36.x34()";
    EXPECT_FLOAT_EQ(e36invX.x35(),e.x35()) << "e36.x35()";
    EXPECT_FLOAT_EQ(e36invX.x36(),e.x36()) << "e36.x36()";
}

/**************************************************************************
 * Basic Operations for 3rd order Tensor (MandelTensor36)
 *************************************************************************/
TEST(mandel_tensors,Basic3rdOrder36e36pluseqe36p){

    TestData td;
    const auto & e36p = td.e36p;
    const auto & e36 = td.e36;
    p3a::mandel3x6 e = e36;
    e+=e36p;

    EXPECT_FLOAT_EQ(e36.x11()+e36p.x11(),e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()+e36p.x12(),e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()+e36p.x13(),e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()+e36p.x14(),e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()+e36p.x15(),e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()+e36p.x16(),e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()+e36p.x21(),e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()+e36p.x22(),e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()+e36p.x23(),e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()+e36p.x24(),e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()+e36p.x25(),e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()+e36p.x26(),e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()+e36p.x31(),e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()+e36p.x32(),e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()+e36p.x33(),e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()+e36p.x34(),e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()+e36p.x35(),e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()+e36p.x36(),e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36e36add36p){

    TestData td;
    const auto & e36p = td.e36p;
    const auto & e36 = td.e36;
    p3a::mandel3x6 e = e36 + e36p;

    EXPECT_FLOAT_EQ(e36.x11()+e36p.x11(),e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()+e36p.x12(),e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()+e36p.x13(),e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()+e36p.x14(),e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()+e36p.x15(),e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()+e36p.x16(),e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()+e36p.x21(),e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()+e36p.x22(),e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()+e36p.x23(),e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()+e36p.x24(),e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()+e36p.x25(),e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()+e36p.x26(),e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()+e36p.x31(),e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()+e36p.x32(),e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()+e36p.x33(),e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()+e36p.x34(),e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()+e36p.x35(),e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()+e36p.x36(),e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36e36minus36p){

    TestData td;
    const auto & e36p = td.e36p;
    const auto & e36 = td.e36;
    p3a::mandel3x6 e = e36 - e36p;

    EXPECT_FLOAT_EQ(e36.x11()-e36p.x11(),e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()-e36p.x12(),e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()-e36p.x13(),e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()-e36p.x14(),e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()-e36p.x15(),e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()-e36p.x16(),e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()-e36p.x21(),e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()-e36p.x22(),e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()-e36p.x23(),e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()-e36p.x24(),e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()-e36p.x25(),e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()-e36p.x26(),e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()-e36p.x31(),e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()-e36p.x32(),e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()-e36p.x33(),e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()-e36p.x34(),e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()-e36p.x35(),e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()-e36p.x36(),e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36e36xreal){

    TestData td;
    const auto & e36 = td.e36;
    const auto & r1 = td.r1;
    p3a::mandel3x6 e = e36;
    e*=r1;

    EXPECT_FLOAT_EQ(e36.x11()*r1,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()*r1,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()*r1,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()*r1,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()*r1,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()*r1,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()*r1,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()*r1,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()*r1,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()*r1,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()*r1,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()*r1,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()*r1,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()*r1,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()*r1,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()*r1,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()*r1,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()*r1,e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36Binarye36xreal){

    TestData td;
    const auto & e36 = td.e36;
    const auto & r1 = td.r1;
    p3a::mandel3x6 e = e36*r1;

    EXPECT_FLOAT_EQ(e36.x11()*r1,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()*r1,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()*r1,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()*r1,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()*r1,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()*r1,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()*r1,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()*r1,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()*r1,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()*r1,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()*r1,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()*r1,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()*r1,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()*r1,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()*r1,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()*r1,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()*r1,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()*r1,e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36Binaryrealxe36){

    TestData td;
    const auto & e36 = td.e36;
    const auto & r1 = td.r1;
    p3a::mandel3x6 e= r1 * e36;

    EXPECT_FLOAT_EQ(e36.x11()*r1,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()*r1,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()*r1,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()*r1,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()*r1,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()*r1,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()*r1,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()*r1,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()*r1,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()*r1,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()*r1,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()*r1,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()*r1,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()*r1,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()*r1,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()*r1,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()*r1,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()*r1,e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36e36divreal){
    TestData td;
    const auto & e36 = td.e36;
    const auto & r1 = td.r1;
    p3a::mandel3x6 e = e36;
    e/=r1;

    EXPECT_FLOAT_EQ(e36.x11()/r1,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()/r1,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()/r1,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()/r1,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()/r1,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()/r1,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()/r1,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()/r1,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()/r1,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()/r1,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()/r1,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()/r1,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()/r1,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()/r1,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()/r1,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()/r1,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()/r1,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()/r1,e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36Binarye36divreal){
    TestData td;
    const auto & e36 = td.e36;
    const auto & r1 = td.r1;
    p3a::mandel3x6 e = e36 / r1;

    EXPECT_FLOAT_EQ(e36.x11()/r1,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()/r1,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()/r1,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()/r1,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()/r1,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()/r1,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()/r1,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()/r1,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()/r1,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()/r1,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()/r1,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()/r1,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()/r1,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()/r1,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()/r1,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()/r1,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()/r1,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()/r1,e.x36()) << "T.x36()";
}

TEST(mandel_tensors,Basic3rdOrder36e36minuseqe36p){

    TestData td;
    const auto & e36p = td.e36p;
    const auto & e36 = td.e36;
    p3a::mandel3x6 e = e36;
    e-=e36p;

    EXPECT_FLOAT_EQ(e36.x11()-e36p.x11(),e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x12()-e36p.x12(),e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x13()-e36p.x13(),e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x14()-e36p.x14(),e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(e36.x15()-e36p.x15(),e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(e36.x16()-e36p.x16(),e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(e36.x21()-e36p.x21(),e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22()-e36p.x22(),e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x23()-e36p.x23(),e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x24()-e36p.x24(),e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(e36.x25()-e36p.x25(),e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(e36.x26()-e36p.x26(),e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(e36.x31()-e36p.x31(),e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x32()-e36p.x32(),e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33()-e36p.x33(),e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x34()-e36p.x34(),e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(e36.x35()-e36p.x35(),e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(e36.x36()-e36p.x36(),e.x36()) << "T.x36()";
}

/**************************************************************************
 * Linear Algegra Operations for 3rd order Tensor (MandelTensor36)
 *************************************************************************/
TEST(mandel_tensors,LinAlg3rdOrder36Transposee36){

    TestData td;
    const auto & e36 = td.e36;
    p3a::mandel6x3 e = transpose(e36);

    EXPECT_FLOAT_EQ(e36.x11(),e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(e36.x21(),e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(e36.x31(),e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(e36.x12(),e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(e36.x22(),e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(e36.x32(),e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(e36.x13(),e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(e36.x23(),e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(e36.x33(),e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(e36.x14(),e.x41()) << "T.x41()";
    EXPECT_FLOAT_EQ(e36.x24(),e.x42()) << "T.x42()";
    EXPECT_FLOAT_EQ(e36.x34(),e.x43()) << "T.x43()";
    EXPECT_FLOAT_EQ(e36.x15(),e.x51()) << "T.x51()";
    EXPECT_FLOAT_EQ(e36.x25(),e.x52()) << "T.x52()";
    EXPECT_FLOAT_EQ(e36.x35(),e.x53()) << "T.x53()";
    EXPECT_FLOAT_EQ(e36.x16(),e.x61()) << "T.x61()";
    EXPECT_FLOAT_EQ(e36.x26(),e.x62()) << "T.x62()";
    EXPECT_FLOAT_EQ(e36.x36(),e.x63()) << "T.x63()";
}

TEST(mandel_tensors,LinAlg3rdOrder36e36xC){

    TestData td;
    p3a::mandel3x6 e = td.e36*td.C;

    EXPECT_FLOAT_EQ(0.4360318402861528,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(0.9345405556169891,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(1.1294252960227749,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(1.7197078302986348,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(1.3479430797635792,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(1.9283632491846265,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(0.2405601714884583,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(1.3601489642543370,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(1.1651977924898005,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(1.9322848070700853,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(1.8562241288914805,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(1.5746750476550175,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(0.5874503614638639,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(2.2398682875834326,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(2.1734397351857577,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(3.3176192350877933,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(2.9597619531868462,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(3.1532472154198636,e.x36()) << "T.x36()";
}

TEST(mandel_tensors,LinAlg3rdOrder36Vxe36){

    TestData td;
    p3a::mandel3x6 e = td.V*td.e36;

    EXPECT_FLOAT_EQ(0.3185304284465936,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(0.2361092230446556,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(0.2355231336813478,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(0.2417284861372397,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(0.5314586051186603,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(0.4802443010730050,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(1.1893596817424412,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(0.4783014754499950,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(0.6839526055434930,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(0.7246316530324255,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(0.8253436509360010,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(1.0188784876197137,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(0.3723594325967821,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(0.5368235788575713,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(0.3956192828621352,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(0.5402554081049711,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(1.0058471318300359,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(0.7936784286137580,e.x36()) << "T.x36()";
}

TEST(mandel_tensors,LinAlg3rdOrder36TensorVxe36){

    TestData td;
    p3a::mandel3x6 e = td.TV*td.e36;

    EXPECT_FLOAT_EQ(0.3185304284465936,e.x11()) << "T.x11()";
    EXPECT_FLOAT_EQ(0.2361092230446556,e.x12()) << "T.x12()";
    EXPECT_FLOAT_EQ(0.2355231336813478,e.x13()) << "T.x13()";
    EXPECT_FLOAT_EQ(0.2417284861372397,e.x14()) << "T.x14()";
    EXPECT_FLOAT_EQ(0.5314586051186603,e.x15()) << "T.x15()";
    EXPECT_FLOAT_EQ(0.4802443010730050,e.x16()) << "T.x16()";
    EXPECT_FLOAT_EQ(1.1893596817424412,e.x21()) << "T.x21()";
    EXPECT_FLOAT_EQ(0.4783014754499950,e.x22()) << "T.x22()";
    EXPECT_FLOAT_EQ(0.6839526055434930,e.x23()) << "T.x23()";
    EXPECT_FLOAT_EQ(0.7246316530324255,e.x24()) << "T.x24()";
    EXPECT_FLOAT_EQ(0.8253436509360010,e.x25()) << "T.x25()";
    EXPECT_FLOAT_EQ(1.0188784876197137,e.x26()) << "T.x26()";
    EXPECT_FLOAT_EQ(0.3723594325967821,e.x31()) << "T.x31()";
    EXPECT_FLOAT_EQ(0.5368235788575713,e.x32()) << "T.x32()";
    EXPECT_FLOAT_EQ(0.3956192828621352,e.x33()) << "T.x33()";
    EXPECT_FLOAT_EQ(0.5402554081049711,e.x34()) << "T.x34()";
    EXPECT_FLOAT_EQ(1.0058471318300359,e.x35()) << "T.x35()";
    EXPECT_FLOAT_EQ(0.7936784286137580,e.x36()) << "T.x36()";
}

/**************************************************************************
 * Constructor Tests for 3rd order Tensor (MandelTensor63)
 *************************************************************************/
TEST(mandel_tensors,Construct63e63fromListxform){

    //already done in setup
    TestData td;
    const auto & e63 = td.e63;

    EXPECT_FLOAT_EQ(0.4759634421237341,e63.x11()) << "e63.x11()";
    EXPECT_FLOAT_EQ(0.8867644229624673,e63.x12()) << "e63.x12()";
    EXPECT_FLOAT_EQ(0.4135404273124733,e63.x13()) << "e63.x13()";
    EXPECT_FLOAT_EQ(0.8568504365470173,e63.x21()) << "e63.x21()";
    EXPECT_FLOAT_EQ(0.1472947329145138,e63.x22()) << "e63.x22()";
    EXPECT_FLOAT_EQ(0.4477610978101461,e63.x23()) << "e63.x23()";
    EXPECT_FLOAT_EQ(0.7307484706668289,e63.x31()) << "e63.x31()";
    EXPECT_FLOAT_EQ(0.2502091219941188,e63.x32()) << "e63.x32()";
    EXPECT_FLOAT_EQ(0.0617040225205491,e63.x33()) << "e63.x33()";
    EXPECT_FLOAT_EQ(0.4119163419942556,e63.x41()) << "e63.x41()";
    EXPECT_FLOAT_EQ(0.1499018857533562,e63.x42()) << "e63.x42()";
    EXPECT_FLOAT_EQ(0.3357608808746134,e63.x43()) << "e63.x43()";
    EXPECT_FLOAT_EQ(0.3481855982554204,e63.x51()) << "e63.x51()";
    EXPECT_FLOAT_EQ(0.5219168123093268,e63.x52()) << "e63.x52()";
    EXPECT_FLOAT_EQ(0.6767633687457478,e63.x53()) << "e63.x53()";
    EXPECT_FLOAT_EQ(0.1958577356414437,e63.x61()) << "e63.x61()";
    EXPECT_FLOAT_EQ(0.3181944873075155,e63.x62()) << "e63.x62()";
    EXPECT_FLOAT_EQ(0.2417581438483243,e63.x63()) << "e63.x63()";
}

TEST(mandel_tensors,Construct63e63frome63Xform){

    //already done in setup
    TestData td;
    const auto & e63(td.e63);

    EXPECT_FLOAT_EQ(0.4759634421237341,e63.x11()) << "e63.x11()";
    EXPECT_FLOAT_EQ(0.8867644229624673,e63.x12()) << "e63.x12()";
    EXPECT_FLOAT_EQ(0.4135404273124733,e63.x13()) << "e63.x13()";
    EXPECT_FLOAT_EQ(0.8568504365470173,e63.x21()) << "e63.x21()";
    EXPECT_FLOAT_EQ(0.1472947329145138,e63.x22()) << "e63.x22()";
    EXPECT_FLOAT_EQ(0.4477610978101461,e63.x23()) << "e63.x23()";
    EXPECT_FLOAT_EQ(0.7307484706668289,e63.x31()) << "e63.x31()";
    EXPECT_FLOAT_EQ(0.2502091219941188,e63.x32()) << "e63.x32()";
    EXPECT_FLOAT_EQ(0.0617040225205491,e63.x33()) << "e63.x33()";
    EXPECT_FLOAT_EQ(0.4119163419942556,e63.x41()) << "e63.x41()";
    EXPECT_FLOAT_EQ(0.1499018857533562,e63.x42()) << "e63.x42()";
    EXPECT_FLOAT_EQ(0.3357608808746134,e63.x43()) << "e63.x43()";
    EXPECT_FLOAT_EQ(0.3481855982554204,e63.x51()) << "e63.x51()";
    EXPECT_FLOAT_EQ(0.5219168123093268,e63.x52()) << "e63.x52()";
    EXPECT_FLOAT_EQ(0.6767633687457478,e63.x53()) << "e63.x53()";
    EXPECT_FLOAT_EQ(0.1958577356414437,e63.x61()) << "e63.x61()";
    EXPECT_FLOAT_EQ(0.3181944873075155,e63.x62()) << "e63.x62()";
    EXPECT_FLOAT_EQ(0.2417581438483243,e63.x63()) << "e63.x63()";
}

TEST(mandel_tensors,Construct63e63frome63){

    TestData td;
    const auto e63invX = td.e63invX;
    p3a::mandel6x3 f(td.e63);
    f.invMandelXform();

    EXPECT_FLOAT_EQ(e63invX.x11(),f.x11()) << "x11()";
    EXPECT_FLOAT_EQ(e63invX.x12(),f.x12()) << "x12()";
    EXPECT_FLOAT_EQ(e63invX.x13(),f.x13()) << "x13()";
    EXPECT_FLOAT_EQ(e63invX.x21(),f.x21()) << "x21()";
    EXPECT_FLOAT_EQ(e63invX.x22(),f.x22()) << "x22()";
    EXPECT_FLOAT_EQ(e63invX.x23(),f.x23()) << "x23()";
    EXPECT_FLOAT_EQ(e63invX.x31(),f.x31()) << "x31()";
    EXPECT_FLOAT_EQ(e63invX.x32(),f.x32()) << "x32()";
    EXPECT_FLOAT_EQ(e63invX.x33(),f.x33()) << "x33()";
    EXPECT_FLOAT_EQ(e63invX.x41(),f.x41()) << "x41()";
    EXPECT_FLOAT_EQ(e63invX.x42(),f.x42()) << "x42()";
    EXPECT_FLOAT_EQ(e63invX.x43(),f.x43()) << "x43()";
    EXPECT_FLOAT_EQ(e63invX.x51(),f.x51()) << "x51()";
    EXPECT_FLOAT_EQ(e63invX.x52(),f.x52()) << "x52()";
    EXPECT_FLOAT_EQ(e63invX.x53(),f.x53()) << "x53()";
    EXPECT_FLOAT_EQ(e63invX.x61(),f.x61()) << "x61()";
    EXPECT_FLOAT_EQ(e63invX.x62(),f.x62()) << "x62()";
    EXPECT_FLOAT_EQ(e63invX.x63(),f.x63()) << "x63()";
}

TEST(mandel_tensors,Construct63e63fromList){

    TestData td;
    const auto e63invX = td.e63invX;
    p3a::mandel6x3 f = td.e63;
    f.invMandelXform();

    EXPECT_FLOAT_EQ(e63invX.x11(),f.x11()) << "x11()";
    EXPECT_FLOAT_EQ(e63invX.x12(),f.x12()) << "x12()";
    EXPECT_FLOAT_EQ(e63invX.x13(),f.x13()) << "x13()";
    EXPECT_FLOAT_EQ(e63invX.x21(),f.x21()) << "x21()";
    EXPECT_FLOAT_EQ(e63invX.x22(),f.x22()) << "x22()";
    EXPECT_FLOAT_EQ(e63invX.x23(),f.x23()) << "x23()";
    EXPECT_FLOAT_EQ(e63invX.x31(),f.x31()) << "x31()";
    EXPECT_FLOAT_EQ(e63invX.x32(),f.x32()) << "x32()";
    EXPECT_FLOAT_EQ(e63invX.x33(),f.x33()) << "x33()";
    EXPECT_FLOAT_EQ(e63invX.x41(),f.x41()) << "x41()";
    EXPECT_FLOAT_EQ(e63invX.x42(),f.x42()) << "x42()";
    EXPECT_FLOAT_EQ(e63invX.x43(),f.x43()) << "x43()";
    EXPECT_FLOAT_EQ(e63invX.x51(),f.x51()) << "x51()";
    EXPECT_FLOAT_EQ(e63invX.x52(),f.x52()) << "x52()";
    EXPECT_FLOAT_EQ(e63invX.x53(),f.x53()) << "x53()";
    EXPECT_FLOAT_EQ(e63invX.x61(),f.x61()) << "x61()";
    EXPECT_FLOAT_EQ(e63invX.x62(),f.x62()) << "x62()";
    EXPECT_FLOAT_EQ(e63invX.x63(),f.x63()) << "x63()";
}

/**************************************************************************
 * Basic Operations for 3rd order Tensor (MandelTensor63)
 *************************************************************************/
TEST(mandel_tensors,Basic3rdOrder63e63pluseqe63p){

    TestData td;
    const auto & e63 = td.e63;
    const auto & e63p = td.e63p;
    p3a::mandel6x3 f = e63;
    f+=e63p;

    EXPECT_FLOAT_EQ(e63.x11()+e63p.x11(),f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()+e63p.x12(),f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()+e63p.x13(),f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()+e63p.x21(),f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()+e63p.x22(),f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()+e63p.x23(),f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()+e63p.x31(),f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()+e63p.x32(),f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()+e63p.x33(),f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()+e63p.x41(),f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()+e63p.x42(),f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()+e63p.x43(),f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()+e63p.x51(),f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()+e63p.x52(),f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()+e63p.x53(),f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()+e63p.x61(),f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()+e63p.x62(),f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()+e63p.x63(),f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrder63e63adde63p){

    TestData td;
    const auto & e63 = td.e63;
    const auto & e63p = td.e63p;
    p3a::mandel6x3 f = e63p + e63;

    EXPECT_FLOAT_EQ(e63.x11()+e63p.x11(),f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()+e63p.x12(),f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()+e63p.x13(),f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()+e63p.x21(),f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()+e63p.x22(),f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()+e63p.x23(),f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()+e63p.x31(),f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()+e63p.x32(),f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()+e63p.x33(),f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()+e63p.x41(),f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()+e63p.x42(),f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()+e63p.x43(),f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()+e63p.x51(),f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()+e63p.x52(),f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()+e63p.x53(),f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()+e63p.x61(),f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()+e63p.x62(),f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()+e63p.x63(),f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrder63e63minuse63p){

    TestData td;
    const auto & e63 = td.e63;
    const auto & e63p = td.e63p;
    p3a::mandel6x3 f = e63 - e63p;

    EXPECT_FLOAT_EQ(e63.x11()-e63p.x11(),f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()-e63p.x12(),f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()-e63p.x13(),f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()-e63p.x21(),f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()-e63p.x22(),f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()-e63p.x23(),f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()-e63p.x31(),f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()-e63p.x32(),f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()-e63p.x33(),f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()-e63p.x41(),f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()-e63p.x42(),f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()-e63p.x43(),f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()-e63p.x51(),f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()-e63p.x52(),f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()-e63p.x53(),f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()-e63p.x61(),f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()-e63p.x62(),f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()-e63p.x63(),f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrder63e63xReal){

    TestData td;
    const auto & e63 = td.e63;
    const auto & r1 = td.r1;
    p3a::mandel6x3 f = e63;
    f*=r1;

    EXPECT_FLOAT_EQ(e63.x11()*r1,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()*r1,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()*r1,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()*r1,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()*r1,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()*r1,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()*r1,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()*r1,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()*r1,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()*r1,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()*r1,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()*r1,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()*r1,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()*r1,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()*r1,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()*r1,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()*r1,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()*r1,f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrderBinary63e63xReal){

    TestData td;
    const auto & e63 = td.e63;
    const auto & r1 = td.r1;
    p3a::mandel6x3 f = e63*r1;

    EXPECT_FLOAT_EQ(e63.x11()*r1,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()*r1,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()*r1,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()*r1,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()*r1,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()*r1,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()*r1,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()*r1,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()*r1,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()*r1,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()*r1,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()*r1,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()*r1,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()*r1,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()*r1,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()*r1,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()*r1,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()*r1,f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrder63BinaryRealxe63){

    TestData td;
    const auto & e63 = td.e63;
    const auto & r1 = td.r1;
    p3a::mandel6x3 f = r1*e63;

    EXPECT_FLOAT_EQ(e63.x11()*r1,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()*r1,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()*r1,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()*r1,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()*r1,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()*r1,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()*r1,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()*r1,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()*r1,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()*r1,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()*r1,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()*r1,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()*r1,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()*r1,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()*r1,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()*r1,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()*r1,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()*r1,f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrder63e63divReal){

    TestData td;
    const auto & e63 = td.e63;
    const auto & r1 = td.r1;
    p3a::mandel6x3 f = e63;
    f/=r1;

    EXPECT_FLOAT_EQ(e63.x11()/r1,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()/r1,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()/r1,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()/r1,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()/r1,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()/r1,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()/r1,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()/r1,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()/r1,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()/r1,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()/r1,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()/r1,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()/r1,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()/r1,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()/r1,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()/r1,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()/r1,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()/r1,f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrder63Binarye63divReal){

    TestData td;
    const auto & e63 = td.e63;
    const auto & r1 = td.r1;
    p3a::mandel6x3 f = e63/r1;

    EXPECT_FLOAT_EQ(e63.x11()/r1,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()/r1,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()/r1,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()/r1,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()/r1,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()/r1,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()/r1,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()/r1,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()/r1,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()/r1,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()/r1,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()/r1,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()/r1,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()/r1,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()/r1,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()/r1,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()/r1,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()/r1,f.x63()) << "e.x63()";
}

TEST(mandel_tensors,Basic3rdOrder63e63minuseqe63p){

    TestData td;
    const auto & e63 = td.e63;
    const auto & e63p = td.e63p;
    p3a::mandel6x3 f=e63;
    f-=e63p;

    EXPECT_FLOAT_EQ(e63.x11()-e63p.x11(),f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x12()-e63p.x12(),f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x13()-e63p.x13(),f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x21()-e63p.x21(),f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22()-e63p.x22(),f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x23()-e63p.x23(),f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x31()-e63p.x31(),f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x32()-e63p.x32(),f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33()-e63p.x33(),f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x41()-e63p.x41(),f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(e63.x42()-e63p.x42(),f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(e63.x43()-e63p.x43(),f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(e63.x51()-e63p.x51(),f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(e63.x52()-e63p.x52(),f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(e63.x53()-e63p.x53(),f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(e63.x61()-e63p.x61(),f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(e63.x62()-e63p.x62(),f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(e63.x63()-e63p.x63(),f.x63()) << "e.x63()";
}

/*************************************************************************
 * Linear Algebra Tests for 3rd order Tensor (MandelTensor63)
 ************************************************************************/
TEST(mandel_tensors,LinAlg3rdOrder63Transposee63){

    TestData td;
    const auto & e63 = td.e63;
    p3a::mandel3x6 f = transpose(e63);

    EXPECT_FLOAT_EQ(e63.x11(),f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(e63.x21(),f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(e63.x31(),f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(e63.x41(),f.x14()) << "e.x14()";
    EXPECT_FLOAT_EQ(e63.x51(),f.x15()) << "e.x15()";
    EXPECT_FLOAT_EQ(e63.x61(),f.x16()) << "e.x16()";
    EXPECT_FLOAT_EQ(e63.x12(),f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(e63.x22(),f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(e63.x32(),f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(e63.x42(),f.x24()) << "e.x24()";
    EXPECT_FLOAT_EQ(e63.x52(),f.x25()) << "e.x25()";
    EXPECT_FLOAT_EQ(e63.x62(),f.x26()) << "e.x26()";
    EXPECT_FLOAT_EQ(e63.x13(),f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(e63.x23(),f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(e63.x33(),f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(e63.x43(),f.x34()) << "e.x34()";
    EXPECT_FLOAT_EQ(e63.x53(),f.x35()) << "e.x35()";
    EXPECT_FLOAT_EQ(e63.x63(),f.x36()) << "e.x36()";
}

TEST(mandel_tensors,LinAlg3rdOrder63Cxe63){

    TestData td;
    p3a::mandel6x3 f = td.C*td.e63;

    EXPECT_FLOAT_EQ(1.7583802475541608,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(1.6477108574850361,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(1.5945079123658290,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(1.5006121644806141,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(1.4573393031354889,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(1.5407967086859855,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(1.6098213949562987,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(0.7956299846332452,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(0.9513912394229351,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(2.0476653342536513,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(0.9335047788801307,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(1.0975390055553489,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(3.0519768882895737,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(1.9588761419308951,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(2.5463107147298092,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(1.9996090051011508,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(1.1079031185693664,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(1.4976255324092389,f.x63()) << "e.x63()";
}

TEST(mandel_tensors,LinAlg3rdOrder63e63xTensorV){

    TestData td;
    p3a::mandel6x3 f = td.e63*td.TV;

    EXPECT_FLOAT_EQ(0.8906847120906268,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(0.9270169652403459,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(1.1081099064471629,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(0.4184921249677286,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(1.1558347277735985,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(0.4153324072539976,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(0.4573911817523391,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(0.6956508580416848,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(0.2819785203466282,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(0.2662789282452787,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(0.6844967313795165,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(0.3486545551228658,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(0.5495077493692450,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(1.0201726332039649,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(0.9132822350759278,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(0.3286483672357691,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(0.4433849126439117,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(0.4539125712913405,f.x63()) << "e.x63()";
}

TEST(mandel_tensors,LinAlg3rdOrder63e63xV6){

    TestData td;
    p3a::mandel6x3 f = td.e63*td.V;

    EXPECT_FLOAT_EQ(0.8906847120906268,f.x11()) << "e.x11()";
    EXPECT_FLOAT_EQ(0.9270169652403459,f.x12()) << "e.x12()";
    EXPECT_FLOAT_EQ(1.1081099064471629,f.x13()) << "e.x13()";
    EXPECT_FLOAT_EQ(0.4184921249677286,f.x21()) << "e.x21()";
    EXPECT_FLOAT_EQ(1.1558347277735985,f.x22()) << "e.x22()";
    EXPECT_FLOAT_EQ(0.4153324072539976,f.x23()) << "e.x23()";
    EXPECT_FLOAT_EQ(0.4573911817523391,f.x31()) << "e.x31()";
    EXPECT_FLOAT_EQ(0.6956508580416848,f.x32()) << "e.x32()";
    EXPECT_FLOAT_EQ(0.2819785203466282,f.x33()) << "e.x33()";
    EXPECT_FLOAT_EQ(0.2662789282452787,f.x41()) << "e.x41()";
    EXPECT_FLOAT_EQ(0.6844967313795165,f.x42()) << "e.x42()";
    EXPECT_FLOAT_EQ(0.3486545551228658,f.x43()) << "e.x43()";
    EXPECT_FLOAT_EQ(0.5495077493692450,f.x51()) << "e.x51()";
    EXPECT_FLOAT_EQ(1.0201726332039649,f.x52()) << "e.x52()";
    EXPECT_FLOAT_EQ(0.9132822350759278,f.x53()) << "e.x53()";
    EXPECT_FLOAT_EQ(0.3286483672357691,f.x61()) << "e.x61()";
    EXPECT_FLOAT_EQ(0.4433849126439117,f.x62()) << "e.x62()";
    EXPECT_FLOAT_EQ(0.4539125712913405,f.x63()) << "e.x63()";
}
