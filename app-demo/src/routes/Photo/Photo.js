import React from 'react';
import {Modal,NavBar} from 'antd-mobile';

export default class Photo extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            modal:JSON.parse(localStorage.getItem("tipShow")) || false
        }
    }
    onClose = key => () => {
        this.setState({
          [key]: true,
        });
    }
    photo(e) {
        
        // 获取当前选中的文件
        if (e) {
            let file = e.target.files[0];
            // debugger
            // console.log(file,e.target.value)
            // 检查文件类型
            // // 图片压缩之旅
            // if (file.type) {
            //     this.props.dispatch({
            //         type: 'picture/updateState',
            //         payload: { file: file }
            //     })
            // }
            // this.transformFileToDataUrl(file);
            
            this.props.history.push({pathname:'/Cropper',query:{file:file}})
        }

    }
    render() {
        return (
            <div className="bg ">
                <div className="widthFix">
                    <NavBar
                        mode="light"
                        // icon={<Icon type="left" />}
                        // onLeftClick={() => console.log('onLeftClick')}
                        rightContent={[
                            <i className="am-icon book" key={'book'} onClick={()=>{this.props.history.push({pathname:'/Baike'})}}></i>
                        ]}
                        >拍照识别</NavBar>
                        <div className="discribe-info">
                            <p>病虫害智能识别</p>
                            <div className="leaf"></div>
                        </div>
                    
                        <div className="btn-box" onClick={() => this.photo()}>
                            <label>
                                <input className="picture" type="file" name="image" onChange={(e) => this.photo(e)} />
                                <div className="choose-btn"><img className="choose-btn-add" src={require("../../assets/photo.png")} alt="logo"/></div>
                            </label>
                        </div>
                    </div>
                    
                
                <Modal 
                    style={{minWidth: '90%'}}
                    visible={!this.state.modal}
                    transparent
                    maskClosable={false}
                    onClose={this.onClose('modal')}
                    title="如何提高识别准确率"   
                    footer={[{ text: '不再提示', onPress: () => {this.onClose('modal')(); localStorage.setItem("tipShow","true");console.log(JSON.parse(localStorage.getItem("tipShow")))} },
                    { text: '我明白了', onPress: () => {this.onClose('modal')(); } }]}
                    wrapProps={{ onTouchStart: this.onWrapTouchStart }}
                    // afterClose={() => { alert('afterClose'); }}
                    >
                    <div style={{overflow: 'auto'}}>
                        请对准作物的叶子拍照<br/>
                        保持良好的光照，并保持一段距离<br/><br/>
                        示例图片：<br/>
                    {/* <div className="leafExp"></div> */}
                    <img src={require("../../assets/leaf.jpg")} alt="示例图片"></img>
                    </div>
                </Modal>
            </div>
        )
    }
}